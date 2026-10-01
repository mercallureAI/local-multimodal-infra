//! Base-mode page preprocessing, as upstream `infer`: EXIF-orient, fit the
//! page into the square with `PIL.ImageOps.pad` (bicubic, centred, padded with
//! the mean colour) and normalize with mean = std = 0.5.

use image::{imageops, DynamicImage, ImageDecoder, ImageReader, Rgb, RgbImage};
use local_error::{InfraError, Result};
use std::path::Path;

/// Pad colour: `int(0.5 * 255)` per channel.
const PAD: u8 = 127;

pub fn load_page(path: &Path) -> Result<RgbImage> {
    let reader = ImageReader::open(path)
        .and_then(|reader| reader.with_guessed_format())
        .map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| InfraError::BadRequest(format!("decode image {}: {e}", path.display())))?;
    let orientation = decoder.orientation().ok();
    let mut image = DynamicImage::from_decoder(decoder)
        .map_err(|e| InfraError::BadRequest(format!("decode image {}: {e}", path.display())))?;
    if let Some(orientation) = orientation {
        image.apply_orientation(orientation);
    }
    let image = image.to_rgb8();
    if image.width() == 0 || image.height() == 0 {
        return Err(InfraError::BadRequest(format!(
            "image {} has no pixels",
            path.display()
        )));
    }
    Ok(image)
}

/// `[3, size, size]` CHW floats in [-1, 1].
pub fn page_tensor(image: &RgbImage, size: u32) -> Vec<f32> {
    let (width, height) = pad_size(image.width(), image.height(), size);
    let resized = imageops::resize(image, width, height, imageops::FilterType::CatmullRom);
    let mut canvas = RgbImage::from_pixel(size, size, Rgb([PAD; 3]));
    let x = round_half_even(f64::from(size - width) * 0.5);
    let y = round_half_even(f64::from(size - height) * 0.5);
    imageops::replace(&mut canvas, &resized, x, y);

    let plane = (size * size) as usize;
    let mut tensor = vec![0.0f32; 3 * plane];
    for (i, pixel) in canvas.pixels().enumerate() {
        for c in 0..3 {
            tensor[c * plane + i] = f32::from(pixel[c]) / 127.5 - 1.0;
        }
    }
    tensor
}

/// The size `ImageOps.pad` resizes to: the long side fills the square.
fn pad_size(width: u32, height: u32, size: u32) -> (u32, u32) {
    let ratio = f64::from(width) / f64::from(height);
    if ratio > 1.0 {
        (
            size,
            (round_half_even(f64::from(size) / ratio) as u32).max(1),
        )
    } else {
        (
            (round_half_even(f64::from(size) * ratio) as u32).max(1),
            size,
        )
    }
}

/// Python's `round`.
fn round_half_even(value: f64) -> i64 {
    let floor = value.floor();
    let diff = value - floor;
    let rounded = if diff > 0.5 || (diff == 0.5 && floor % 2.0 != 0.0) {
        floor + 1.0
    } else {
        floor
    };
    rounded as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_size_fills_the_long_side() {
        assert_eq!(pad_size(2000, 1000, 1024), (1024, 512));
        assert_eq!(pad_size(1000, 2000, 1024), (512, 1024));
        assert_eq!(pad_size(300, 300, 1024), (1024, 1024));
        // Small pages are scaled up, not stretched.
        assert_eq!(pad_size(400, 100, 1024), (1024, 256));
    }

    #[test]
    fn round_matches_python() {
        assert_eq!(round_half_even(0.5), 0);
        assert_eq!(round_half_even(1.5), 2);
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(2.6), 3);
    }

    #[test]
    fn page_tensor_pads_with_the_mean_colour() {
        let image = RgbImage::from_pixel(100, 50, Rgb([255, 255, 255]));
        let tensor = page_tensor(&image, 64);
        let plane = 64 * 64;
        assert_eq!(tensor.len(), 3 * plane);
        // Top-left corner is padding, the centre is the white page.
        assert!((tensor[0] - (127.0 / 127.5 - 1.0)).abs() < 1e-6);
        assert!((tensor[32 * 64 + 32] - 1.0).abs() < 1e-6);
    }
}
