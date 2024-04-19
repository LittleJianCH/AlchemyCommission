#![feature(associated_type_defaults)]
#![feature(const_refs_to_static)]

mod types;
mod parser;
mod model;

use constcat::concat;
use nalgebra::SVector;
use std::{fs::File, io::Read};

use types::*;
use model::*;

const DATASET_DIR: &str = "../../datasets/MNIST/raw/";
const TEST_IMAGES: &str = concat!(DATASET_DIR, "t10k-images-idx3-ubyte");
const TEST_LABELS: &str = concat!(DATASET_DIR, "t10k-labels-idx1-ubyte");

fn test(model: impl Model<IMAGE_SIZE, 10>) {
    let data_set = parser::parse_data_set(TEST_IMAGES, TEST_LABELS);

    let mut correct = 0;
    let mut total = 0;

    for (image, label) in data_set {
        let predict = model.forward(&image);

        let max_index = predict.argmax().0;

        if max_index == label as usize {
            correct += 1;
        }
        total += 1;

        if total % 100 == 0 {
            println!("Total: {}, accuracy: {}", total, correct as f64 / total as f64);
        }
    }

    println!("Accuracy: {}", correct as f64 / total as f64);
}

pub type MnistModel = ConcatModel<
    IMAGE_SIZE, 256, 10,
    ConcatModel<
        IMAGE_SIZE, 256, 256,
        LinearModel<IMAGE_SIZE, 256>,
        ReLUModel<256>
    >,
    ConcatModel<
        256, 20, 10,
        ConcatModel<
            256, 20, 20,
            LinearModel<256, 20>,
            ReLUModel<20>
        >,
        LinearModel<20, 10>
    >
>;

fn main() {
    let mut s = String::new();

    File::open("model.json").expect("Could not create model file")
          .read_to_string(&mut s).expect("Could not read model file");
    
    let new_model: MnistModel = serde_json::from_str(s.as_str()).unwrap();

    test(new_model);
}
