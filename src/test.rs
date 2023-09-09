extern crate image;
extern crate ndarray;

use image::{GenericImageView, GenericImage, ImageBuffer, Luma, DynamicImage};
use image::imageops::{grayscale, invert};
use image::imageops::colorops::Contrast;
use image::imageops::FilterType;
use ndarray::{Array2, ArrayView2};
use std::f64::consts::PI;


struct StringArtGenerator {
    iterations: u32,
    shape: String,
    image: Option<DynamicImage>,
    data: Option<Array2<f64>>,
    residual: Option<Array2<f64>>,
    seed: u32,
    nails: u32,
    weight: f64,
    nodes: Vec<(f64, f64)>,
    paths: Vec<Vec<Vec<(usize, usize)>>>,
}

impl StringArtGenerator {
    fn new() -> Self {
        Self {
            iterations: 1000,
            shape: "circle".to_string(),
            image: None,
            data: None,
            residual: None,
            seed: 0,
            nails: 100,
            weight: 20.0,
            nodes: Vec::new(),
            paths: Vec::new(),
        }
    }

    fn set_seed(&mut self, seed: u32) {
        self.seed = seed;
    }

    fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }

    fn set_shape(&mut self, shape: &str) {
        self.shape = shape.to_string();
    }

    fn set_nails(&mut self, nails: u32) {
        self.nails = nails;
        if self.shape == "circle" {
            self.set_nodes_circle();
        } else if self.shape == "rectangle" {
            self.set_nodes_rectangle();
        }
    }

    fn set_iterations(&mut self, iterations: u32) {
        self.iterations = iterations;
    }

    fn set_nodes_rectangle(&mut self) {
        let perimeter = self.get_perimeter();
        let spacing = perimeter / (self.nails as f64);
        let (width, height) = self.data.as_ref().unwrap().dim();

        let pnails = (0..self.nails).map(|t| t as f64 * spacing).collect::<Vec<_>>();

        let mut xarr = Vec::new();
        let mut yarr = Vec::new();
        for p in pnails {
            let (x, y) = if p < width as f64 {
                (p, 0.0)
            } else if p < (width + height) as f64 {
                (width as f64, p - width as f64)
            } else if p < (2.0 * width + height) as f64 {
                (width as f64 - (p - width as f64 - height as f64), height as f64)
            } else {
                (0.0, height as f64 - (p - 2.0 * width as f64 - height as f64))
            };
            xarr.push(x);
            yarr.push(y);
        }

        self.nodes = xarr.into_iter().zip(yarr.into_iter()).collect();
    }

    fn get_perimeter(&self) -> f64 {
        2.0 * (self.data.as_ref().unwrap().dim().0 + self.data.as_ref().unwrap().dim().1) as f64
    }

    fn set_nodes_circle(&mut self) {
        let spacing = 2.0 * PI / (self.nails as f64);

        let steps = 0..self.nails;

        let radius = self.get_radius();

        let x = steps.map(|t| radius + radius * ((t as f64 * spacing).cos())).collect::<Vec<_>>();
        let y = steps.map(|t| radius + radius * ((t as f64 * spacing).sin())).collect::<Vec<_>>();

        self.nodes = x.into_iter().zip(y.into_iter()).collect();
    }

    fn get_radius(&self) -> f64 {
        0.5 * (self.data.as_ref().unwrap().dim().0.min(self.data.as_ref().unwrap().dim().1)) as f64
    }

    fn load_image(&mut self, path: &str) {
        let img = image::open(path).unwrap();
        self.image = Some(img.clone());
        let np_img = ArrayView2::from_shape((img.width() as usize, img.height() as usize), img.as_bytes()).unwrap().to_owned();
        self.data = Some(np_img);
    }

    fn preprocess(&mut self) {
        let img = grayscale(self.image.as_ref().unwrap());
        let img = invert(&img);
        let img = img.filter3x3(&[
            -1.0, -1.0, -1.0,
            -1.0, 8.0, -1.0,
            -1.0, -1.0, -1.0,
        ]);
        let img = Contrast::new(img).contrast(1.0);
        let np_img = ArrayView2::from_shape((img.width() as usize, img.height() as usize), img.as_bytes()).unwrap().to_owned();
        self.data = Some(np_img);
    }

    fn generate(&mut self) -> Vec<(f64, f64)> {
        self.calculate_paths();
        let mut delta = 0.0;
        let mut pattern = Vec::new();
        let mut nail = self.seed;
        let datacopy = self.data.as_ref().unwrap().to_owned();
        for _ in 0..self.iterations {
            let (darkest_nail, darkest_path) = self.choose_darkest_path(nail);

            pattern.push(self.nodes[darkest_nail]);

            self.data.as_mut().unwrap().zip_mut_with(&darkest_path, |x, y| *x -= self.weight * y);
            self.data.as_mut().unwrap().mapv_inplace(|x| if x < 0.0 { 0.0 } else { x });

            if self.data.as_ref().unwrap().sum() <= 0.0 {
                println!("Stopping iterations. No more data or residual unchanged.");
                break;
            }

            delta = self.data.as_ref().unwrap().sum();

            nail = darkest_nail;
        }

        self.residual = Some(self.data.as_ref().unwrap().to_owned());
        self.data = Some(datacopy);

        pattern
    }

    fn choose_darkest_path(&self, nail: u32) -> (usize, Array2<f64>) {
        let mut max_darkness = -1.0;
        let mut darkest_nail = 0;
        let mut darkest_path = Array2::zeros(self.data.as_ref().unwrap().dim());
        for (index, rowcol) in self.paths[nail as usize].iter().enumerate() {
            let rows = rowcol.iter().map(|x| x.0).collect::<Vec<_>>();
            let cols = rowcol.iter().map(|x| x.1).collect::<Vec<_>>();
            let darkness = self.data.as_ref().unwrap()[&rows, &cols].sum();

            if darkness > max_darkness {
                darkest_path.fill(0.0);
                for (row, col) in rowcol {
                    darkest_path[[*row, *col]] = 1.0;
                }
                darkest_nail = index;
                max_darkness = darkness;
            }
        }

        (darkest_nail, darkest_path)
    }

    fn calculate_paths(&mut self) {
        for (nail, anode) in self.nodes.iter().enumerate() {
            self.paths.push(Vec::new());
            for node in &self.nodes {
                let path = bresenham_path(*anode, *node, self.data.as_ref().unwrap().dim());
                self.paths[nail].push(path);
            }
        }
    }
}

fn bresenham_path(start: (f64, f64), end: (f64, f64), shape: (usize, usize)) -> Vec<(usize, usize)> {
    let (mut x1, mut y1) = start;
    let (mut x2, mut y2) = end;

    x1 = x1.max(0.0).min((shape.0 - 1) as f64).round();
    y1 = y1.max(0.0).min((shape.1 - 1) as f64).round();
    x2 = x2.max(0.0).min((shape.0 - 1) as f64).round();
    y2 = y2.max(0.0).min((shape.1 - 1) as f64).round();

    let dx = x2 - x1;
    let dy = y2 - y1;

    let mut path = Vec::new();

    if start == end {
        return path;
    }

    let is_steep = dy.abs() > dx.abs();

    if is_steep {
        std::mem::swap(&mut x1, &mut y1);
        std::mem::swap(&mut x2, &mut y2);
    }

    if x1 > x2 {
        std::mem::swap(&mut x1, &mut x2);
        std::mem::swap(&mut y1, &mut y2);
    }

    let dx = x2 - x1;
    let dy = y2 - y1;

    let error = (dx / 2.0) as i32;
    let ystep = if y1 < y2 { 1 } else { -1 };

    let mut y = y1 as i32;
    for x in x1 as i32..=x2 as i32 {
        if is_steep {
            path.push((y as usize, x as usize));
        } else {
            path.push((x as usize, y as usize));
        }
        let mut error = error - dy.abs() as i32;
        if error < 0 {
            y += ystep;
            error += dx as i32;
        }
    }

    path
}
