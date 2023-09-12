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
                for color in palette {
                    res.push(((color & 0xff0000) >> 16) as u8);
                    res.push(((color & 0x00ff00) >> 8) as u8);
                    res.push((color & 0x0000ff) as u8);
                }
                res
            };
            gif::Encoder::new(
                std::fs::File::create(&path_).unwrap(),
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
        let half: T = 0.5_f64.as_();
        let iwmin: i64 = (x - rad - half).ceil().as_();
        let iwmax: i64 = (x + rad - half).floor().as_();
        let ihmin: i64 = (y - rad - half).ceil().as_();
        let ihmax: i64 = (y + rad - half).floor().as_();
        for ih in ihmin..ihmax + 1 {
            if ih < 0 || ih >= self.height.try_into().unwrap() { continue; }
            for iw in iwmin..iwmax + 1 {
                if iw < 0 || iw >= self.width.try_into().unwrap() { continue; }
                let w: T = iw.as_() + half; // pixel center
                let h: T = ih.as_() + half; // pixel center
                if (w - x) * (w - x) + (h - y) * (h - y) > rad * rad { continue; }
                let idata = (self.height - 1 - ih as usize) * self.width + iw as usize;
                self.data[idata] = color;
            }
        }
    }

    pub fn paint_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, rad: f32, color: u8) {

        let (iwmin,iwmax,ihmin,ihmax) = {
            let iwmin0 = (x0 - rad - 0.5_f32).ceil() as i64;
            let iwmax0 = (x0 + rad - 0.5_f32).floor() as i64;
            let ihmin0 = (y0 - rad - 0.5_f32).ceil() as i64;
            let ihmax0 = (y0 + rad - 0.5_f32).floor() as i64;
            let iwmin1 = (x1 - rad - 0.5_f32).ceil() as i64;
            let iwmax1 = (x1 + rad - 0.5_f32).floor() as i64;
            let ihmin1 = (y1 - rad - 0.5_f32).ceil() as i64;
            let ihmax1 = (y1 + rad - 0.5_f32).floor() as i64;
            (
                std::cmp::min(iwmin0, iwmin1),
                std::cmp::max(iwmax0, iwmax1),
                std::cmp::min(ihmin0, ihmin1),
                std::cmp::max(ihmax0, ihmax1))
        };
        let sqlen = (x1-x0) * (x1-x0) + (y1-y0) * (y1-y0);
        for ih in ihmin..ihmax + 1 {
            if ih < 0 || ih >= self.height.try_into().unwrap() { continue; }
            for iw in iwmin..iwmax + 1 {
                if iw < 0 || iw >= self.width.try_into().unwrap() { continue; }
                let w = iw as f32 + 0.5_f32; // pixel center
                let h = ih as f32 + 0.5_f32; // pixel center
                let t = ((w-x0)*(x1-x0) + (h-y0)*(y1-y0))/sqlen;
                let sqdist = if t < 0. {
                    (w-x0)*(w-x0) + (h-y0)*(h-y0)
                } else if t > 1. {
                    (w-x1)*(w-x1) + (h-y1)*(h-y1)
                } else {
                    (w-x0)*(w-x0) + (h-y0)*(h-y0) - sqlen * t * t
                };
                if sqdist > rad * rad { continue; }
                let idata = (self.height - 1 - ih as usize) * self.width + iw as usize;
                self.data[idata] = color;
            }
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
