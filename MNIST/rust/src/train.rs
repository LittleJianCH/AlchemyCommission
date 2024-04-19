#![feature(associated_type_defaults)]
#![feature(const_refs_to_static)]

mod types;
mod parser;
mod model;

use constcat::concat;
use nalgebra::SVector;
use std::{fs::File, io::Write};

use types::*;
use model::*;

const DATASET_DIR: &str = "../../datasets/MNIST/raw/";
const TRAINING_IMAGES: &str = concat!(DATASET_DIR, "train-images-idx3-ubyte");
const TRAINING_LABELS: &str = concat!(DATASET_DIR, "train-labels-idx1-ubyte");

fn mid_layer<const N: usize, const M: usize>() -> impl Model<N, M> {
    ConcatModel::new(
        LinearModel::new(),
        ReLUModel::new()
    )
}

fn initial_model() -> impl Model<IMAGE_SIZE, 10> {
    ConcatModel::new(
        mid_layer(),
        ConcatModel::new(
            mid_layer::<256, 20>(),
            LinearModel::new()
        )
    )
}

fn train() -> impl Model<IMAGE_SIZE, 10> {
    let model = initial_model();

    let mut loss = Loss { m: model };

    let data_set = parser::parse_data_set(TRAINING_IMAGES, TRAINING_LABELS);

    let mut total = 0;
    for (image, label) in data_set {
        let ys = 
            SVector::<f64, 10>::from_fn(|i, _| if i == label as usize { 1.0 } else { 0.0 });
        
        let _predict = loss.m.forward(&image);
        loss.adjust(0.01, &ys, &image);

        total += 1;    
        if total % 100 == 0 {
            println!("Total: {}", total);
        }
    }

    loss.m
}

fn main() {
    let model = train();

    let json = serde_json::to_string(&model).unwrap();

    File::create("model.json").expect("Could not create model file")
         .write_all(json.as_bytes()).expect("Could not write model file");

    println!("model-size: {} bytes", json.len());
}
