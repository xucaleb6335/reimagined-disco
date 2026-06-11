//VGA pixel buffer helpers for the DE1-SoC UP VGA subsystem
//(Option A of docs/vga_demo_plan.md)
//
//The pixel buffer is 320x240 @ 16bpp RGB565 with the UP IP's addressing:
//byte offset = (y << 10) | (x << 1), i.e. a 1 KB line stride.
//VGA_PIXBUF_PHYS below is the Qsys-assigned base as seen from the HPS
//(here: on-chip pixel buffer behind the h2f bridge window) — record the
//real assignment from Qsys next to ACCEL_OFFSET, do not guess.

#ifndef VGA_H
#define VGA_H

#include <stdint.h>

#define VGA_WIDTH   320
#define VGA_HEIGHT  240
#define VGA_Y_SHIFT 10               //1 KB line stride
#define VGA_SPAN    (VGA_HEIGHT << VGA_Y_SHIFT)

#define VGA_PIXBUF_PHYS 0xC8000000u  //h2f bridge + Qsys pixel buffer base

#define VGA_RGB(r, g, b) ((uint16_t)((((r) & 0x1F) << 11) | \
                                     (((g) & 0x3F) << 5)  | \
                                      ((b) & 0x1F)))

#define VGA_BLACK   VGA_RGB( 0,  0,  0)
#define VGA_WHITE   VGA_RGB(31, 63, 31)
#define VGA_RED     VGA_RGB(31,  0,  0)
#define VGA_GREEN   VGA_RGB( 0, 63,  0)
#define VGA_BLUE    VGA_RGB( 0,  0, 31)
#define VGA_YELLOW  VGA_RGB(31, 63,  0)
#define VGA_CYAN    VGA_RGB( 0, 63, 31)
#define VGA_MAGENTA VGA_RGB(31,  0, 31)
#define VGA_GREY    VGA_RGB(12, 24, 12)

#define VGA_FONT_W 8
#define VGA_FONT_H 8

//mmap the pixel buffer through /dev/mem; returns 0 on success
int vga_open(void);
void vga_close(void);

//Point the helpers at an arbitrary VGA_SPAN-sized buffer instead
//(bare-metal, or off-screen rendering in tests)
void vga_attach(void *framebuffer);

void vga_pixel(int x, int y, uint16_t color);
void vga_fill_rect(int x, int y, int w, int h, uint16_t color);
void vga_clear(uint16_t color);

//Diverging heatmap for int32 accumulators: negative = blue..cyan,
//positive = red..yellow..white, 0 = black; vmax scales full intensity
uint16_t vga_heatmap(int32_t v, int32_t vmax);

//8x8 font (uppercase letters, digits, basic punctuation; lowercase is
//mapped to uppercase, unsupported glyphs render as space)
void vga_putc(int x, int y, char ch, uint16_t fg, uint16_t bg);
void vga_text(int x, int y, const char *s, uint16_t fg, uint16_t bg);

//Bring-up test pattern (plan step A.0): 8 full-height color bars
void vga_color_bars(void);

#endif
