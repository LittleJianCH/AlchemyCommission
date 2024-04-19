#![feature(associated_type_defaults)]
#![feature(const_refs_to_static)]

mod types;
mod parser;
mod model;

use constcat::concat;
use nalgebra::SVector;

use types::*;
use model::*;

const DATASET_DIR: &str = "../../datasets/MNIST/raw/";
const TRAINING_IMAGES: &str = concat!(DATASET_DIR, "train-images-idx3-ubyte");
const TRAINING_LABELS: &str = concat!(DATASET_DIR, "train-labels-idx1-ubyte");
const TEST_IMAGES: &str = concat!(DATASET_DIR, "t10k-images-idx3-ubyte");
const TEST_LABELS: &str = concat!(DATASET_DIR, "t10k-labels-idx1-ubyte");

fn mid_layer<const N: usize, const M: usize>() -> Box<dyn Model<N, M>> {
    Box::new(ConcatModel::new(
        Box::new(LinearModel::new()),
        Box::new(ReLUModel::new())
    ))
}

fn initial_model() -> Box<dyn Model<IMAGE_SIZE, 10>> {
    Box::new(ConcatModel::new(
        mid_layer() as Box<dyn Model<IMAGE_SIZE, 512>>,
        Box::new(ConcatModel::new(
            mid_layer() as Box<dyn Model<512, 128>>,
            Box::new(LinearModel::new()) as Box<dyn Model<128, 10>>
        ))
    ))
}

fn train() -> Box<dyn Model<IMAGE_SIZE, 10>> {
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

fn test(model: Box<dyn Model<IMAGE_SIZE, 10>>) {
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

fn main() {
    let model = train();

    test(model);
}
