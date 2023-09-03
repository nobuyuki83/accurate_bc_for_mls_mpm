pub struct Canvas {
    pub width: usize,
    pub height: usize,
    data: Vec<u8>,
    path: std::path::PathBuf,
    gif_enc: Option<gif::Encoder<std::fs::File>>,
}

impl Canvas {
    pub fn new(path_: &std::path::Path, size: (usize, usize)) -> Self {
        Self {
            width: size.0,
            height: size.1,
            data: vec!(0; size.0 * size.1 * 3),
            path: path_.try_into().unwrap(),
            gif_enc: None,
        }
    }

    pub fn paint_circle(&mut self, x: f32, y: f32, rad: f32, color: i32) {
        let iwmin = (x - rad).floor().clamp(0_f32, self.width as f32) as i64;
        let iwmax = (x + rad).floor().clamp(0_f32, self.width as f32) as i64;
        let ihmin = (y - rad).floor().clamp(0_f32, self.width as f32) as i64;
        let ihmax = (y + rad).floor().clamp(0_f32, self.width as f32) as i64;
        let r = ((color & 0xff0000) >> 16) as u8;
        let g = ((color & 0x00ff00) >> 8) as u8;
        let b = (color & 0x0000ff) as u8;
        for iw in iwmin..iwmax + 1 {
            for ih in ihmin..ihmax + 1 {
                if iw < 0 || iw >= self.width.try_into().unwrap() { continue; }
                if ih < 0 || ih >= self.height.try_into().unwrap() { continue; }
                let w = iw as f32;
                let h = ih as f32;
                if (w - x) * (w - x) + (h - y) * (h - y) > rad * rad { continue; }
                let idata = (self.height - 1 - ih as usize) * self.width + iw as usize;
                self.data[idata * 3 + 0] = r;
                self.data[idata * 3 + 1] = g;
                self.data[idata * 3 + 2] = b;
            }
        }
    }

    pub fn clear(&mut self, color: i32) {
        let r = ((color & 0xff0000) >> 16) as u8;
        let g = ((color & 0x00ff00) >> 8) as u8;
        let b = (color & 0x0000ff) as u8;
        for ih in 0..self.height {
            for iw in 0..self.width {
                self.data[(ih * self.width + iw) * 3 + 0] = r;
                self.data[(ih * self.width + iw) * 3 + 1] = g;
                self.data[(ih * self.width + iw) * 3 + 2] = b;
            }
        }
    }

    pub fn write(&mut self) {
        // For reading and opening files
        dbg!(&self.path);
        let file = std::fs::File::create(&self.path).unwrap();
        let ref mut w = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(
            w,
            self.width.try_into().unwrap(),
            self.height.try_into().unwrap()); // Width is 2 pixels and height is 1.
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&self.data).unwrap(); // Save
    }
}


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

    pub fn paint_circle(&mut self, x: f32, y: f32, rad: f32, color: u8) {
        let iwmin = (x - rad).floor().clamp(0_f32, self.width as f32) as i64;
        let iwmax = (x + rad).floor().clamp(0_f32, self.width as f32) as i64;
        let ihmin = (y - rad).floor().clamp(0_f32, self.width as f32) as i64;
        let ihmax = (y + rad).floor().clamp(0_f32, self.width as f32) as i64;
        for iw in iwmin..iwmax + 1 {
            for ih in ihmin..ihmax + 1 {
                if iw < 0 || iw >= self.width.try_into().unwrap() { continue; }
                if ih < 0 || ih >= self.height.try_into().unwrap() { continue; }
                let w = iw as f32;
                let h = ih as f32;
                if (w - x) * (w - x) + (h - y) * (h - y) > rad * rad { continue; }
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
