extern crate image;
extern crate ndarray;

use image::DynamicImage;
use image::imageops::invert;
use ndarray::{Axis, Array2, ArrayView2};
use std::{f64::consts::PI, time::Instant, fs};
use imageproc::contrast::stretch_contrast;
use crate::image::EncodableLayout;
use plotters::prelude::*;


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

fn main() {
    let start = Instant::now();

    let mut generator = StringArtGenerator::new();
    generator.load_image(r"C:\Users\Kaustubh Karthik\Documents\Computer_Science\Rust_Projects\string_art\src\images\input\stickman.jpg");
    generator.preprocess();
    generator.set_nails(10);
    generator.set_seed(5);
    generator.set_iterations(50);
    let pattern = generator.generate();

    let root = BitMapBackend::new(r"C:\Users\Kaustubh Karthik\Documents\Computer_Science\Rust_Projects\string_art\src\images\output\string_stickman.jpg", (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(0)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0f32..1f32, 0f32..1f32)
        .unwrap();

    chart.configure_mesh().disable_x_mesh().disable_y_mesh().draw().unwrap();

    for i in 0..pattern.len()-1 {
        chart.draw_series(LineSeries::new(
            vec![(pattern[i].0 as f32, pattern[i].1 as f32), (pattern[i+1].0 as f32, pattern[i+1].1 as f32)],
            &BLACK,
        )).unwrap();
    }

    root.present().unwrap();

    let duration = start.elapsed();
    println!("Time elapsed in expensive_function() is: {:?}", duration);
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
            nails: 50,
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
            } else if p < (2.0 * (width + height) as f64) as f64 {
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

        let _steps: std::ops::Range<u32> = 0..self.nails;

        let radius = self.get_radius();

        let x = (0..self.nails).map(|t| radius + radius * ((t as f64 * spacing).cos())).collect::<Vec<_>>();
        let y = (0..self.nails).map(|t| radius + radius * ((t as f64 * spacing).sin())).collect::<Vec<_>>();
        

        self.nodes = x.into_iter().zip(y.into_iter()).collect();
    }

    fn get_radius(&self) -> f64 {
        0.5 * (self.data.as_ref().unwrap().dim().0.min(self.data.as_ref().unwrap().dim().1)) as f64
    }

    fn load_image(&mut self, path: &str) {
        let img = image::open(path).unwrap();
        self.image = Some(img.clone());
        let np_img = ArrayView2::from_shape((img.width() as usize, img.height() as usize), img.as_bytes()).unwrap().to_owned();
        let np_img_f64 = np_img.mapv(|x| x as f64);
        self.data = Some(np_img_f64);

    }

    fn preprocess(&mut self) {
        let mut img = self.image.as_ref().unwrap().to_luma8();
        invert(&mut img);
        let img = stretch_contrast(&img, 0, 255);
        let np_img = ArrayView2::from_shape((img.width() as usize, img.height() as usize), img.as_bytes())
            .unwrap()
            .to_owned()
            .mapv(|x| x as f64);
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

            nail = darkest_nail as u32;
        }

        self.residual = Some(self.data.as_ref().unwrap().to_owned());
        self.data = Some(datacopy);

        pattern
    }

    fn choose_darkest_path(&self, nail: u32) -> (usize, Array2<f64>) {
        let mut max_darkness = -1.0;
        let mut darkest_nail = 0;
        let mut darkest_path = Array2::zeros(self.data.as_ref().unwrap().dim());
        let (n_rows, n_cols) = self.data.as_ref().unwrap().dim();
        for (index, rowcol) in self.paths[nail as usize].iter().enumerate() {
            let rows: Vec<usize> = rowcol.iter().map(|x| x.0.min(n_rows - 1)).collect();
            let cols: Vec<usize> = rowcol.iter().map(|x| x.1.min(n_cols - 1)).collect();
            let darkness = self.data.as_ref().unwrap().select(Axis(0), &rows).select(Axis(1), &cols).sum();
            if darkness > max_darkness {
                darkest_path.fill(0.0);
                for (row, col) in rowcol {
                    darkest_path[[*row.min(&(n_rows - 1)), *col.min(&(n_cols - 1))]] = 1.0;
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
