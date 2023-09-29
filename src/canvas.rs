use num_traits::AsPrimitive;

pub struct Canvas {
    pub width: usize,
    pub height: usize,
    data: Vec<u8>,
}

pub fn pixels_in_circle<Real>(
    x: Real, y: Real, rad: Real,
    width: usize, height: usize) -> Vec<usize>
    where Real: num_traits::Float + 'static + AsPrimitive<i64>,
          i64: AsPrimitive<Real>,
          f64: AsPrimitive<Real>
{
    let half: Real = 0.5_f64.as_();
    let iwmin: i64 = (x - rad - half).ceil().as_();
    let iwmax: i64 = (x + rad - half).floor().as_();
    let ihmin: i64 = (y - rad - half).ceil().as_();
    let ihmax: i64 = (y + rad - half).floor().as_();
    /*
    let iwmin: i64 = (x - rad).floor().as_();
    let iwmax: i64 = (x + rad).floor().as_();
    let ihmin: i64 = (y - rad).floor().as_();
    let ihmax: i64 = (y + rad).floor().as_();
     */
    let mut res = Vec::<usize>::new();
    for iw in iwmin..iwmax + 1 {
        if iw < 0 || iw >= width.try_into().unwrap() { continue; }
        for ih in ihmin..ihmax + 1 {
            if ih < 0 || ih >= height.try_into().unwrap() { continue; }
            let w: Real = iw.as_() + half; // pixel center
            let h: Real = ih.as_() + half; // pixel center
//            let w: Real = iw.as_();
//            let h: Real = ih.as_();
            if (w - x) * (w - x) + (h - y) * (h - y) > rad * rad { continue; }
            let idata = (height - 1 - ih as usize) * width + iw as usize;
            res.push(idata);
        }
    }
    res
}

pub fn pixels_in_line(
    x0: f32, y0: f32,
    x1: f32, y1: f32, rad: f32,
    width: usize, height: usize) -> Vec<usize> {
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
    let mut res = Vec::<usize>::new();
    for ih in ihmin..ihmax + 1 {
        if ih < 0 || ih >= height.try_into().unwrap() { continue; }
        for iw in iwmin..iwmax + 1 {
            if iw < 0 || iw >= width.try_into().unwrap() { continue; }
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
            let idata = (height - 1 - ih as usize) * width + iw as usize;
            res.push(idata);
        }
    }
    res
}

pub fn rgb(color: i32) -> (u8, u8, u8) {
    let r = ((color & 0xff0000) >> 16) as u8;
    let g = ((color & 0x00ff00) >> 8) as u8;
    let b = (color & 0x0000ff) as u8;
    (r,g,b)
}

impl Canvas {
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            width: size.0,
            height: size.1,
            data: vec!(0; size.0 * size.1 * 3),
        }
    }

    pub fn paint_circle<Real>(&mut self, x: Real, y: Real, rad: Real, color: i32)
    where Real: num_traits::Float + 'static + AsPrimitive<i64>,
          i64: AsPrimitive<Real>,
          f64: AsPrimitive<Real>
    {
        let (r,g,b) = rgb(color);
        let pixs = pixels_in_circle(x,y,rad,self.width, self.height);
        for idata in pixs {
            self.data[idata * 3 + 0] = r;
            self.data[idata * 3 + 1] = g;
            self.data[idata * 3 + 2] = b;
        }
    }

    pub fn paint_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, rad: f32, color: i32) {
        let (r,g,b) = rgb(color);
        let pixs = pixels_in_line(x0,y0, x1, y1, rad, self.width, self.height);
        for idata in pixs {
            self.data[idata * 3 + 0] = r;
            self.data[idata * 3 + 1] = g;
            self.data[idata * 3 + 2] = b;
        }
    }

    pub fn clear(&mut self, color: i32) {
        let (r,g,b) = rgb(color);
        for ih in 0..self.height {
            for iw in 0..self.width {
                self.data[(ih * self.width + iw) * 3 + 0] = r;
                self.data[(ih * self.width + iw) * 3 + 1] = g;
                self.data[(ih * self.width + iw) * 3 + 2] = b;
            }
        }
    }

    pub fn write(&mut self, path_: &std::path::Path) {
        // For reading and opening files
        let file = std::fs::File::create(path_).unwrap();
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

