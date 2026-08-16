/*
   ============================================================================
   Author: Serhii Trush(Techn0man1ac)
   Project: Entropy-Image-Prioritization
   Repository: https://github.com/techn0man1ac/Entropy-Image-Prioritization
   Description: On-Board Joint RGB Entropy Analysis & ROI Selection for ESP32-S3
   ============================================================================
*/

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <JPEGDEC.h>
#include <math.h>
#include <algorithm>

// ============================================================================
// 1. CONFIGURATION & MEMORY CONSTRAINTS (Entropy-Image-Prioritization)
// ============================================================================
const char* ssid = "your-ssid"
const char* password = "your-password"
const char* urlPicture = "https://raw.githubusercontent.com/techn0man1ac/Entropy-Image-Prioritization/refs/heads/main/Python/MilkaCat_baseline.jpg"; // JPG baseline!

#define BLOCK_SIZE 50            // ROI block dimension
#define TOP_PERCENT 10.0f        // Percentage of top high-entropy blocks to prioritize
#define ALPHA 0.35f              // Heatmap transparency overlay coefficient

#define IMG_SCALE_WIDTH 8        // Image width scale (8 - defalt)
#define IMG_SCALE_HEIGHT 8       // Image height scale (8 - defalt)

#define MAX_IMAGE_WIDTH 512
#define MAX_IMAGE_HEIGHT 512
#define MAX_BLOCKS ((MAX_IMAGE_WIDTH / BLOCK_SIZE + 1) * (MAX_IMAGE_HEIGHT / BLOCK_SIZE + 1))
#define MAX_JPEG_BUF_SIZE (512 * 1024)

typedef struct {
  int x, y, w, h;
  float entropy;
  int priority_rank;
} ImageBlock;

// Static pointers allocated in PSRAM to prevent run-time heap fragmentation
uint8_t *jpg_buf = NULL;
uint8_t *rgb888_buf = NULL;
ImageBlock *blocks_buf = NULL;

uint16_t img_width = 0;
uint16_t img_height = 0;
int total_blocks = 0;

WebServer server(80);
JPEGDEC jpeg;

// ============================================================================
// 2. JPEG DECODER DRAW CALLBACK
// ============================================================================
int JPEGDraw(JPEGDRAW *pDraw) {
  if (!rgb888_buf) return 0;

  uint16_t *bitmap = pDraw->pPixels;
  int block_w = pDraw->iWidth;
  int block_h = pDraw->iHeight;

  for (int j = 0; j < block_h; j++) {
    int py = pDraw->y + j;
    if (py >= img_height) continue;

    for (int i = 0; i < block_w; i++) {
      int px = pDraw->x + i;
      if (px >= img_width) continue;

      int dst_idx = (py * img_width + px) * 3;
      uint16_t color = bitmap[j * block_w + i];

      // Extract RGB565 components
      uint8_t r = (uint8_t)((color >> 11) & 0x1F);
      uint8_t g = (uint8_t)((color >> 5)  & 0x3F);
      uint8_t b = (uint8_t)(color         & 0x1F);

      // Accurate conversion to RGB888
      rgb888_buf[dst_idx]     = (r * 527 + 23) >> 6;
      rgb888_buf[dst_idx + 1] = (g * 259 + 33) >> 6;
      rgb888_buf[dst_idx + 2] = (b * 527 + 23) >> 6;
    }
  }
  return 1;
}

// ============================================================================
// 3. JOINT RGB ENTROPY CALCULATION H(R,G,B) [Python Equivalence]
// ============================================================================
float calculate_block_joint_rgb_entropy(const uint8_t *rgb_pixels, int x0, int y0, int block_w, int block_h, int img_w) {
  int num_pixels = block_w * block_h;
  if (num_pixels == 0) return 0.0f;

  // Static histogram to prevent stack overflow on Xtensa architecture
  static uint16_t histogram[65536];
  memset(histogram, 0, sizeof(histogram));

  for (int y = 0; y < block_h; y++) {
    for (int x = 0; x < block_w; x++) {
      int px = (y0 + y) * img_w + (x0 + x);
      uint8_t r = rgb_pixels[px * 3];
      uint8_t g = rgb_pixels[px * 3 + 1];
      uint8_t b = rgb_pixels[px * 3 + 2];

      // Quantize colors into 16-bit RGB565 buckets
      uint16_t color16 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
      histogram[color16]++;
    }
  }

  float entropy = 0.0f;
  float inv_pixels = 1.0f / (float)num_pixels;

  for (int i = 0; i < 65536; i++) {
    if (histogram[i] > 0) {
      float p = (float)histogram[i] * inv_pixels;
      entropy -= p * log2f(p);
    }
  }

  return entropy;
}

// ============================================================================
// 4. JET COLORMAP GENERATOR
// ============================================================================
void get_jet_color(float normalized_val, uint8_t &r, uint8_t &g, uint8_t &b) {
  float v = normalized_val * 4.0f;

  float rf = fminf(fmaxf(fminf(v - 1.5f, -v + 4.5f), 0.0f), 1.0f);
  float gf = fminf(fmaxf(fminf(v - 0.5f, -v + 3.5f), 0.0f), 1.0f);
  float bf = fminf(fmaxf(fminf(v + 0.5f, -v + 2.5f), 0.0f), 1.0f);

  r = (uint8_t)(rf * 255.0f);
  g = (uint8_t)(gf * 255.0f);
  b = (uint8_t)(bf * 255.0f);
}

// ============================================================================
// 5. MAIN PIPELINE & DETAILED LOGGING (Zero Run-Time Malloc)
// ============================================================================
void process_image_from_web() {
  Serial.println("\n=================================================");
  Serial.println("[EIP Core] Starting Image Prioritization Pipeline...");
  Serial.printf("[EIP Memory] Free DRAM: %d bytes | Free PSRAM: %d bytes\n", ESP.getFreeHeap(), ESP.getFreePsram());

  WiFiClientSecure secureClient;
  secureClient.setInsecure();
  secureClient.setHandshakeTimeout(15000);

  HTTPClient http;
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  http.setTimeout(15000);

  Serial.printf("[EIP Network] Downloading target frame from: %s\n", urlPicture);

  if (http.begin(secureClient, urlPicture)) {
    int httpCode = http.GET();
    if (httpCode == HTTP_CODE_OK) {
      WiFiClient *stream = http.getStreamPtr();
      int bytes_read = 0;
      unsigned long start_time = millis();
      unsigned long last_read = millis();

      while (http.connected() && (bytes_read < MAX_JPEG_BUF_SIZE) && (millis() - start_time < 30000)) {
        size_t available = stream->available();
        if (available > 0) {
          int c = stream->readBytes(&jpg_buf[bytes_read], available);
          if (c > 0) {
            bytes_read += c;
            last_read = millis();
          }
        } else {
          if (bytes_read > 0 && (millis() - last_read > 3000)) break;
          delay(1);
          yield();
        }
      }

      Serial.printf("[EIP Network] Successfully read %d bytes in %lu ms\n", bytes_read, millis() - start_time);

      if (bytes_read > 0) {
        jpeg.setPixelType(RGB565_LITTLE_ENDIAN);

        if (jpeg.openRAM(jpg_buf, bytes_read, JPEGDraw)) {
          uint16_t orig_w = jpeg.getWidth();
          uint16_t orig_h = jpeg.getHeight();

          int scale = JPEG_SCALE_EIGHTH;

          img_width = orig_w / IMG_SCALE_WIDTH;
          img_height = orig_h / IMG_SCALE_HEIGHT;

          if (img_width > MAX_IMAGE_WIDTH) img_width = MAX_IMAGE_WIDTH;
          if (img_height > MAX_IMAGE_HEIGHT) img_height = MAX_IMAGE_HEIGHT;

          Serial.printf("[EIP Decoder] Original size: %dx%d | Target processing size: %dx%d px\n", orig_w, orig_h, img_width, img_height);

          unsigned long decode_start = millis();
          jpeg.decode(0, 0, scale);
          Serial.printf("[EIP Decoder] Frame decoded successfully in %lu ms\n", millis() - decode_start);

          total_blocks = 0;
          float min_ent = 999.0f, max_ent = -999.0f;

          Serial.println("\n--- STARTING BLOCK-BY-BLOCK ENTROPY ANALYSIS ---");

          // Compute Joint RGB Entropy across grid blocks
          for (int y = 0; y < img_height; y += BLOCK_SIZE) {
            for (int x = 0; x < img_width; x += BLOCK_SIZE) {
              if (total_blocks >= MAX_BLOCKS) break;

              int bw = (x + BLOCK_SIZE > img_width) ? (img_width - x) : BLOCK_SIZE;
              int bh = (y + BLOCK_SIZE > img_height) ? (img_height - y) : BLOCK_SIZE;

              blocks_buf[total_blocks].x = x;
              blocks_buf[total_blocks].y = y;
              blocks_buf[total_blocks].w = bw;
              blocks_buf[total_blocks].h = bh;

              float ent = calculate_block_joint_rgb_entropy(rgb888_buf, x, y, bw, bh, img_width);
              blocks_buf[total_blocks].entropy = ent;
              blocks_buf[total_blocks].priority_rank = 0;

              // Detailed logging for each scanned block
              Serial.printf("[Block #%02d] X:%3d | Y:%3d | Size:%dx%d | Joint Entropy: %.4f bits/pixel\n",
                            total_blocks, x, y, bw, bh, ent);

              if (ent < min_ent) min_ent = ent;
              if (ent > max_ent) max_ent = ent;

              total_blocks++;
            }
          }
          Serial.println("--- END OF BLOCK-BY-BLOCK ENTROPY ANALYSIS ---\n");
          Serial.printf("[EIP Stats] Min Entropy: %.4f | Max Entropy: %.4f\n", min_ent, max_ent);

          int selected_count = std::max(1, (int)ceilf(total_blocks * (TOP_PERCENT / 100.0f)));

          // Deterministic partial sort O(N log K) to find highest entropy priorities
          std::partial_sort(blocks_buf, blocks_buf + selected_count, blocks_buf + total_blocks,
          [](const ImageBlock & a, const ImageBlock & b) {
            return a.entropy > b.entropy;
          });

          Serial.printf("\n--- TOP %d%% SELECTED HIGH-ENTROPY REGIONS (%d blocks) ---\n", (int)TOP_PERCENT, selected_count);
          for (int i = 0; i < selected_count; i++) {
            blocks_buf[i].priority_rank = i + 1;
            Serial.printf("[Priority Rank #%02d] X:%3d, Y:%3d | Entropy: %.4f\n",
                          blocks_buf[i].priority_rank, blocks_buf[i].x, blocks_buf[i].y, blocks_buf[i].entropy);
          }
          Serial.println("----------------------------------------------------\n");

          float ent_range = (max_ent - min_ent > 0.001f) ? (max_ent - min_ent) : 1.0f;

          // Apply Jet Heatmap Overlay & Grid Rendering
          for (int b = 0; b < total_blocks; b++) {
            int x = blocks_buf[b].x;
            int y = blocks_buf[b].y;
            int w = blocks_buf[b].w;
            int h = blocks_buf[b].h;
            int x2 = x + w - 1;
            int y2 = y + h - 1;

            float norm_ent = (blocks_buf[b].entropy - min_ent) / ent_range;
            uint8_t jr, jg, jb;
            get_jet_color(norm_ent, jr, jg, jb);
            bool is_priority = (blocks_buf[b].priority_rank > 0);

            for (int cy = y; cy <= y2; cy++) {
              for (int cx = x; cx <= x2; cx++) {
                int pidx = (cy * img_width + cx) * 3;
                bool is_border = (cx == x || cx == x2 || cy == y || cy == y2);

                if (is_border) {
                  if (is_priority) {
                    // Red border for prioritized top regions
                    rgb888_buf[pidx]     = 255;
                    rgb888_buf[pidx + 1] = 0;
                    rgb888_buf[pidx + 2] = 0;
                  } else {
                    // White grid boundary for regular blocks
                    rgb888_buf[pidx]     = 255;
                    rgb888_buf[pidx + 1] = 255;
                    rgb888_buf[pidx + 2] = 255;
                  }
                } else {
                  // Jet Colormap blending overlay
                  rgb888_buf[pidx]     = (uint8_t)(rgb888_buf[pidx] * (1.0f - ALPHA) + jr * ALPHA);
                  rgb888_buf[pidx + 1] = (uint8_t)(rgb888_buf[pidx + 1] * (1.0f - ALPHA) + jg * ALPHA);
                  rgb888_buf[pidx + 2] = (uint8_t)(rgb888_buf[pidx + 2] * (1.0f - ALPHA) + jb * ALPHA);
                }
              }
            }
          }
          jpeg.close();
        }
      }
    }
    http.end();
  }

  Serial.printf("[EIP Core] Pipeline Complete. Free PSRAM: %d bytes\n", ESP.getFreePsram());
  Serial.println("=================================================\n");
}

// ============================================================================
// 6. HTTP SERVER & BMP STREAMING MODULE
// ============================================================================
void handle_image_bmp() {
  if (!rgb888_buf) {
    server.send(500, "text/plain", "Frame buffer is uninitialized");
    return;
  }

  uint32_t row_padding = (4 - ((img_width * 3) % 4)) % 4;
  uint32_t row_stride = (img_width * 3) + row_padding;
  uint32_t filesize = 54 + (row_stride * img_height);
  int32_t h = -img_height;

  uint8_t bmp_header[54] = {
    'B', 'M',
    (uint8_t)(filesize & 0xFF), (uint8_t)((filesize >> 8) & 0xFF), (uint8_t)((filesize >> 16) & 0xFF), (uint8_t)((filesize >> 24) & 0xFF),
    0, 0, 0, 0,
    54, 0, 0, 0,
    40, 0, 0, 0,
    (uint8_t)(img_width & 0xFF), (uint8_t)((img_width >> 8) & 0xFF), (uint8_t)((img_width >> 16) & 0xFF), (uint8_t)((img_width >> 24) & 0xFF),
    (uint8_t)(h & 0xFF), (uint8_t)((h >> 8) & 0xFF), (uint8_t)((h >> 16) & 0xFF), (uint8_t)((h >> 24) & 0xFF),
    1, 0, 24, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  WiFiClient client = server.client();
  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.setContentLength(filesize);
  server.send(200, "image/bmp", "");

  client.write(bmp_header, 54);

  static uint8_t row_buf[MAX_IMAGE_WIDTH * 3 + 4];
  memset(row_buf, 0, row_stride);

  for (int y = 0; y < img_height; y++) {
    for (int x = 0; x < img_width; x++) {
      int idx = (y * img_width + x) * 3;
      row_buf[x * 3 + 0] = rgb888_buf[idx + 2]; // B
      row_buf[x * 3 + 1] = rgb888_buf[idx + 1]; // G
      row_buf[x * 3 + 2] = rgb888_buf[idx + 0]; // R
    }
    client.write(row_buf, row_stride);
  }
}

void handle_root() {
  String html = "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Entropy Image Prioritization Dashboard</title></head>";
  html += "<body style='text-align:center; background:#111827; color:#f3f4f6; font-family:sans-serif; margin:0; padding:20px;'>";
  html += "<h1 style='color:#60a5fa;'>Entropy-Image-Prioritization</h1>";
  html += "<p style='color:#9ca3af;'>On-Board Joint RGB Entropy Analysis & ROI Selection for ESP32-S3</p>";
  html += "<div style='display:inline-block; border:1px solid #374151; padding:10px; background:#1f2937; border-radius:8px;'>";
  html += "<img src='/image.bmp?t=" + String(millis()) + "' style='max-width:100%; height:auto; border-radius:4px;'/>";
  html += "</div>";
  html += "<p style='font-size:12px; color:#6b7280; margin-top:15px;'>Highlighted red borders represent top high-entropy regions.</p>";
  html += "</body></html>";
  server.send(200, "text/html; charset=utf-8", html);
}

// ============================================================================
// 7. SETUP & PSRAM STATIC ALLOCATION
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // Check PSRAM availability and allocate static buffers once
  if (psramFound()) {
    Serial.printf("[EIP System] PSRAM detected! Size: %d bytes\n", ESP.getPsramSize());

    jpg_buf = (uint8_t *)ps_malloc(MAX_JPEG_BUF_SIZE);
    rgb888_buf = (uint8_t *)ps_malloc(MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * 3);
    blocks_buf = (ImageBlock *)ps_malloc(sizeof(ImageBlock) * MAX_BLOCKS);

    if (!jpg_buf || !rgb888_buf || !blocks_buf) {
      Serial.println("[CRITICAL ERROR] Failed to allocate static buffers in PSRAM!");
      while (1) {
        delay(1000);
      }
    }
    Serial.println("[EIP System] Static PSRAM buffers allocated successfully.");
  } else {
    Serial.println("[CRITICAL ERROR] PSRAM is missing or disabled in Arduino IDE settings!");
    while (1) {
      delay(1000);
    }
  }

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n[EIP System] WiFi connected successfully.");
  Serial.print("[EIP System] Web Server IP: http://");
  Serial.println(WiFi.localIP());

  // Execute processing pipeline
  process_image_from_web();

  server.on("/", handle_root);
  server.on("/image.bmp", handle_image_bmp);
  server.begin();
}

void loop() {
  server.handleClient();
}
