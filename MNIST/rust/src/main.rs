mod types;
mod parser;

use constcat::concat;

const DATASET_DIR: &str = "../../datasets/MNIST/raw/";
const TRAINING_IMAGES: &str = concat!(DATASET_DIR, "train-images-idx3-ubyte");
const TRAINING_LABELS: &str = concat!(DATASET_DIR, "train-labels-idx1-ubyte");

fn main() {
    let data_set = parser::parse_data_set(TRAINING_IMAGES, TRAINING_LABELS);

    println!("Number of images: {}", data_set.len());
    println!("First image: {:?}", data_set[0]);
    println!("Last image: {:?}", data_set[data_set.len() - 1]);
}
