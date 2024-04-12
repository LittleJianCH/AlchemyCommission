use nalgebra::{SVector, SMatrix};

type Vector<const N: usize> = SVector<f64, N>;
type Matrix<const N: usize, const M: usize> = SMatrix<f64, N, M>;

pub trait Model<
    const N: usize, const M: usize
> where
{
    fn forward(&self, input: &Vector<N>) -> Vector<M>;
    fn graident(&self, input: &Vector<N>) -> Matrix<M, N>;
    fn adjust(&mut self, mu: f64, gra: &Matrix<1, M>, input: &Vector<N>);
}


pub struct LinearModel<
    const N: usize, const M: usize
> {
    pub weights: SMatrix<f64, M, N>,
    pub bias: SVector<f64, M>,
}

impl<const N: usize, const M: usize> Model<N, M> for LinearModel<N, M> {
    fn forward(&self, input: &SVector<f64, N>) -> SVector<f64, M> {
        self.weights * input + self.bias
    }

    fn graident(&self, _input: &SVector<f64, N>) -> SMatrix<f64, M, N> {
        self.weights.clone()
    }

    fn adjust(&mut self, mu: f64, gra: &Matrix<1, M>, input: &Vector<N>) {
        self.bias -= mu * gra.transpose();
        self.weights -= mu * (input * gra).transpose();
    }
}


pub struct ReLUModel<const N: usize> {}

impl<const N: usize> Model<N, N> for ReLUModel<N> {
    fn forward(&self, input: &SVector<f64, N>) -> SVector<f64, N> {
        input.map(|x| x.max(0.0))
    }

    fn graident(&self, input: &SVector<f64, N>) -> SMatrix<f64, N, N> {
        SMatrix::from_diagonal(
            &input.map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        )
    }

    fn adjust(&mut self, _mu: f64, _gra: &Matrix<1, N>, _input: &Vector<N>) {}
}


pub struct ConcatModel<
    const N: usize, const M: usize, const O: usize,
> {
    m1: Box<dyn Model<N, M>>,
    m2: Box<dyn Model<M, O>>,
}

impl<
    const N: usize, const M: usize, const O: usize,
> Model<N, O> for ConcatModel<N, M, O> {
    fn forward(&self, input: &Vector<N>) -> Vector<O> {
        self.m2.forward(&self.m1.forward(input))
    }

    fn graident(&self, input: &Vector<N>) -> Matrix<O, N> {
        let m2_input = self.m1.forward(input);

        self.m2.graident(&m2_input) * self.m1.graident(input)
    }

    fn adjust(&mut self, mu: f64, gra: &Matrix<1, O>, input: &Vector<N>) {
        let m2_input = self.m1.forward(input);
        let m1_gra = gra * self.m2.graident(&m2_input);

        self.m2.adjust(mu, gra, &m2_input);
        self.m1.adjust(mu, &m1_gra, input);
    }
}

pub struct Loss<const N: usize, const M: usize> {
    m: Box<dyn Model<N, M>>
}

impl<const N: usize, const M: usize> Loss<N, M> {
    fn adjust(&mut self, mu: f64, ys: &Vector<M>, xs: &Vector<N>) {
        let predict = self.m.forward(xs);

        self.m.adjust(mu, &(2.0 * (predict - ys)).transpose(), xs);
    }
}