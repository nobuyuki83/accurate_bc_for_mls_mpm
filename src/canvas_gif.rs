use num_traits::AsPrimitive;

pub struct CanvasGif {
    pub width: usize,
    pub height: usize,
    data: Vec<u8>,
    gif_enc: Option<gif::Encoder<std::fs::File>>,
}

impl CanvasGif {
    pub fn new(path_: &std::path::Path,
               size: (usize, usize),
               palette: &Vec<i32>) -> Self {
        let res_encoder = {
            let global_palette = {
                let mut res: Vec<u8> = vec!();
                for &color in palette {
                    let (r, g, b) = crate::canvas::rgb(color);
                    res.push(r);
                    res.push(g);
                    res.push(b);
                }
                res
            };
            gif::Encoder::new(
                std::fs::File::create(path_).unwrap(),
                size.0.try_into().unwrap(),
                size.1.try_into().unwrap(),
                &global_palette)
        };
        match res_encoder {
            Err(_e) => {
                Self {
                    width: size.0,
                    height: size.1,
                    data: vec!(0; size.0 * size.1),
                    gif_enc: None,
                }
            }
            Ok(t) => {
                Self {
                    width: size.0,
                    height: size.1,
                    data: vec!(0; size.0 * size.1),
                    gif_enc: Some(t),
                }
            }
        }
    }

    pub fn paint_point<T>(
        &mut self,
        x: T, y: T, transf: &nalgebra::Matrix3::<T>,
        point_size: T, color: u8)
        where T: num_traits::Float + 'static + AsPrimitive<i64> + nalgebra::RealField,
              f64: AsPrimitive<T>,
              i64: AsPrimitive<T>
    {
        let a = transf * nalgebra::Vector3::<T>::new(x, y, T::one());
        let pixs = crate::canvas::pixels_in_point(
            a.x, a.y,
            point_size, self.width, self.height);
        for idata in pixs {
            self.data[idata] = color;
        }
    }

    #[allow(clippy::identity_op)]
    pub fn paint_polyloop<T>(
        &mut self,
        vtx2xy: &[T],
        transform: &nalgebra::Matrix3::<T>,
        point_size: T, color: u8)
        where T: num_traits::Float + 'static + AsPrimitive<i64> + nalgebra::RealField,
              f64: AsPrimitive<T>,
              i64: AsPrimitive<T>
    {
        let n = vtx2xy.len() / 2;
        for i in 0..n {
            let j = (i + 1) % n;
            self.paint_line(
                vtx2xy[i * 2 + 0], vtx2xy[i * 2 + 1],
                vtx2xy[j * 2 + 0], vtx2xy[j * 2 + 1], transform, point_size, color);
        }
    }

    pub fn paint_line<T>(
        &mut self,
        x0: T, y0: T,
        x1: T, y1: T,
        transform: &nalgebra::Matrix3::<T>,
        rad: T, color: u8)
        where T: num_traits::Float + 'static + AsPrimitive<i64> + nalgebra::RealField,
              f64: AsPrimitive<T>,
              i64: AsPrimitive<T>
    {
        let a0 = transform * nalgebra::Vector3::<T>::new(x0, y0, T::one());
        let a1 = transform * nalgebra::Vector3::<T>::new(x1, y1, T::one());
        let pixs = crate::canvas::pixels_in_line(a0.x, a0.y, a1.x, a1.y, rad, self.width, self.height);
        for idata in pixs {
            self.data[idata] = color;
        }
    }

    pub fn clear(&mut self, color: u8) {
        for ih in 0..self.height {
            for iw in 0..self.width {
                self.data[ih * self.width + iw] = color;
            }
        }
    }

    pub fn write(&mut self) {
        // For reading and opening files
        let mut frame = gif::Frame::default();
        frame.width = self.width as u16;
        frame.height = self.height as u16;
        frame.buffer = std::borrow::Cow::Borrowed(&self.data);
        match &mut self.gif_enc {
            None => {}
            Some(enc) => {
                let _ = &enc.write_frame(&frame);
            }
        }
    }
}
