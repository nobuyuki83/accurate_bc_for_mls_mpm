use num_traits::AsPrimitive;

pub struct CanvasGif {
    pub width: usize,
    pub height: usize,
    data: Vec<u8>,
    gif_enc: Option<gif::Encoder<std::fs::File>>
}

impl CanvasGif {
    pub fn new(path_: &std::path::Path,
               size: (usize, usize),
               palette: &Vec<i32>) -> Self {
        let res_encoder = {
            let global_palette = {
                let mut res: Vec<u8> = vec!();
                for &color in palette {
                    let (r,g,b) = crate::canvas::rgb(color);
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

    pub fn paint_circle<T>(&mut self, x: T, y: T, rad: T, color: u8)
    where T: num_traits::Float + 'static + AsPrimitive<i64>,
    f64: AsPrimitive<T>,
    i64: AsPrimitive<T>
    {
        let pixs = crate::canvas::pixels_in_circle(x,y,rad,self.width, self.height);
        for idata in pixs {
            self.data[idata] = color;
        }
    }

    pub fn paint_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, rad: f32, color: u8) {
        let pixs = crate::canvas::pixels_in_line(x0, y0, x1, y1, rad, self.width, self.height);
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
