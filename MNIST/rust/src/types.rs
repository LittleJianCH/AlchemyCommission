use nalgebra::SVector;

pub const IMAGE_SIZE: usize = 28 * 28;

pub type Image = SVector<f64, IMAGE_SIZE>;

pub type LabeledImage = (Image, u8);

pub type DataSet = Vec<LabeledImage>;
