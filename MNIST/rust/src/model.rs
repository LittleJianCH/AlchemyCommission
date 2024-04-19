use std::cell::Cell;

type Vector<const N: usize> = nalgebra::SVector<f64, N>;
type Matrix<const N: usize, const M: usize> = nalgebra::SMatrix<f64, N, M>;

pub trait Model<
    const N: usize, const M: usize
> where
{
    fn forward(&self, input: &Vector<N>) -> Vector<M>;
    fn forward_cached(&self) -> Vector<M>;
    fn graident(&self, input: &Vector<N>) -> Matrix<M, N>;
    fn adjust(&mut self, mu: f64, gra: &Matrix<1, M>, input: &Vector<N>);
}


pub struct LinearModel<
    const N: usize, const M: usize
> {
    pub weights: Matrix<M, N>,
    pub bias: Vector<M>,
    cache: Cell<Vector<M>>
}

impl<const N: usize, const M: usize> LinearModel<N, M> {
    pub fn new() -> Self {
        Self {
            weights: Matrix::from_fn(|_, _| rand::random::<f64>() / 100.0),
            bias: Vector::from_fn(|_, _| rand::random::<f64>() / 100.0),
            cache: Cell::new(Vector::zeros())
        }
    }
}

impl<const N: usize, const M: usize> Model<N, M> for LinearModel<N, M> {
    fn forward(&self, input: &Vector<N>) -> Vector<M> {
        self.cache.set(self.weights * input + self.bias);
        self.forward_cached()
    }

    fn forward_cached(&self) -> Vector<M> {
        self.cache.get()
    }

    fn graident(&self, _input: &Vector<N>) -> Matrix<M, N> {
        self.weights.clone()
    }

    fn adjust(&mut self, mu: f64, gra: &Matrix<1, M>, input: &Vector<N>) {
        self.bias -= mu * gra.transpose();
        self.weights -= mu * (input * gra).transpose();
    }
}


pub struct ReLUModel<const N: usize> {
    cache: Cell<Vector<N>>
}

impl<const N: usize> ReLUModel<N> {
    pub fn new() -> Self {
        Self {
            cache: Cell::new(Vector::zeros())
        }
    }
}

impl<const N: usize> Model<N, N> for ReLUModel<N> {
    fn forward(&self, input: &Vector<N>) -> Vector<N> {
        self.cache.set(input.map(|x| 
            if x < 0.0 {
                x * 0.05
            } else {
                x
            }
        ));
        self.forward_cached()
    }

    fn forward_cached(&self) -> Vector<N> {
        self.cache.get()
    }

    fn graident(&self, input: &Vector<N>) -> Matrix<N, N> {
        Matrix::from_diagonal(
            &input.map(|x| if x < 0.0 { 0.05 } else { 1.0 })
        )
    }

    fn adjust(&mut self, _mu: f64, _gra: &Matrix<1, N>, _input: &Vector<N>) {}
}


pub struct SigmoidModel<const N: usize> {
    cache: Cell<Vector<N>>
}

impl<const N: usize> SigmoidModel<N> {
    pub fn new() -> Self {
        Self {
            cache: Cell::new(Vector::zeros())
        }
    }

    fn f(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl<const N: usize> Model<N, N> for SigmoidModel<N> {
    fn forward(&self, input: &Vector<N>) -> Vector<N> {
        self.cache.set(input.map(Self::f));
        self.forward_cached()
    }

    fn forward_cached(&self) -> Vector<N> {
        self.cache.get()
    }

    fn graident(&self, _input: &Vector<N>) -> Matrix<N, N> {
        let s = self.forward_cached();

        Matrix::from_diagonal(
            &s.map(|x| Self::f(x) * (1.0 - Self::f(x)))
        )
    }

    fn adjust(&mut self, _mu: f64, _gra: &Matrix<1, N>, _input: &Vector<N>) {}
}


pub struct SoftMaxModel<const N: usize> {
    cache: Cell<Vector<N>>
}

impl<const N: usize> SoftMaxModel<N> {
    pub fn new() -> Self {
        Self {
            cache: Cell::new(Vector::zeros())
        }
    }
}

impl<const N: usize> Model<N, N> for SoftMaxModel<N> {
    fn forward(&self, input: &Vector<N>) -> Vector<N> {
        let exs = input.map(|x| x.exp());
        let s = exs.sum();

        self.cache.set(exs.map(|x| x / s));
        self.forward_cached()
    }

    fn forward_cached(&self) -> Vector<N> {
        self.cache.get()
    }

    fn graident(&self, input: &Vector<N>) -> Matrix<N, N> {
        let exs = input.map(|x| x.exp());
        let s = exs.sum();

        Matrix::from_diagonal(
            &input.map(|x| {
                let ex = x.exp();
                ex * (s - ex) / (s * s)
            })
        )
    }

    fn adjust(&mut self, _mu: f64, _gra: &Matrix<1, N>, _input: &Vector<N>) {}
}


pub struct ConcatModel<
    const N: usize, const M: usize, const O: usize,
> {
    m1: Box<dyn Model<N, M>>,
    m2: Box<dyn Model<M, O>>,
    cache: Cell<Vector<O>>
}

impl<const N: usize, const M: usize, const O: usize> ConcatModel<N, M, O> {
    pub fn new(m1: Box<dyn Model<N, M>>, m2: Box<dyn Model<M, O>>) -> Self {
        Self { m1, m2, cache: Cell::new(Vector::zeros()) }
    }
}

impl<
    const N: usize, const M: usize, const O: usize,
> Model<N, O> for ConcatModel<N, M, O> {
    fn forward(&self, input: &Vector<N>) -> Vector<O> {
        self.cache.set(self.m2.forward(&self.m1.forward(input)));
        self.forward_cached()
    }

    fn forward_cached(&self) -> Vector<O> {
        self.cache.get()
    }

    fn graident(&self, input: &Vector<N>) -> Matrix<O, N> {
        let m2_input = self.m1.forward_cached();

        self.m2.graident(&m2_input) * self.m1.graident(input)
    }

    fn adjust(&mut self, mu: f64, gra: &Matrix<1, O>, input: &Vector<N>) {
        let m2_input = self.m1.forward_cached();
        let m1_gra = gra * self.m2.graident(&m2_input);

        self.m2.adjust(mu, gra, &m2_input);
        self.m1.adjust(mu, &m1_gra, input);
    }
}

pub struct Loss<const N: usize, const M: usize> {
    pub m: Box<dyn Model<N, M>>
}

impl<const N: usize, const M: usize> Loss<N, M> {
    pub fn adjust(&mut self, mu: f64, ys: &Vector<M>, xs: &Vector<N>) {
        let predict = self.m.forward_cached();

        self.m.adjust(mu, &(2.0 * (predict - ys)).transpose(), xs);
    }
}