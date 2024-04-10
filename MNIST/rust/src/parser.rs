use std::fs::File;
use std::io::Read;

use crate::types::*;

pub fn parse_data_set(image_path: &str, label_path: &str) -> DataSet {
    let images = read_images(image_path);
    let labels = read_labels(label_path);

    assert!(images.len() == labels.len(), "Number of images and labels do not match");

    images.into_iter().zip(labels.into_iter()).collect()
}

fn read_images(path: &str) -> Vec<Image> {
    let mut file = File::open(path).expect("Could not open image file");

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Could not read image file");

    assert_eq!(buffer[0..4], [0, 0, 8, 3], "Invalid magic number");

    let count = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
    // row and col are ignored

    (0..count).map(|index| {
        let offset = 16 + index * IMAGE_SIZE;
        Image::from_fn(|i, _| buffer[offset + i] as f64 / 255.0)
    }).collect()
}

fn read_labels(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect("Could not open label file");

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Could not read label file");

    assert_eq!(buffer[0..4], [0, 0, 8, 1], "Invalid magic number");

    let count = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;

    (0..count).map(|index| buffer[8 + index]).collect()
}