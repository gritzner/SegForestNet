use crate::utils::*;
use std::sync::Arc;

fn is_clockwise(polygon: &NdArray<f64>) -> bool {
    let mut area = 0.;
    for i in 0..polygon.shape[0]-1 {
        area += polygon[[i, 0]] * polygon[[i+1, 1]];
        area -= polygon[[i, 1]] * polygon[[i+1, 0]];
    }
    area >= 0.
}

fn is_inside(polygon: &NdArray<f64>, x: f64, y: f64) -> bool {
    let mut crossings: usize = 0;
    
    for i in 0..polygon.shape[0]-1 {
        let xs = [polygon[[i, 0]], polygon[[i+1, 0]]];
        let ys = [polygon[[i, 1]], polygon[[i+1, 1]]];
        
        if y < ys[0].min(ys[1]) || y > ys[0].max(ys[1]) {
            continue;
        }
        
        if ys[0] != ys[1] {
            let mut intersection = (y - ys[0]) / (ys[1] - ys[0]);
            intersection *= xs[1] - xs[0];
            intersection += xs[0];
            crossings += if x <= intersection && intersection <= 2.0 { 1 } else { 0 };
        } else {
            crossings += if x <= xs[0] && xs[0] <= 2.0 { 1 } else { 0 };
            crossings += if x <= xs[1] && xs[1] <= 2.0 { 1 } else { 0 };
        }
    }
    
    (crossings % 2) == 1
}

//PYTHON_EXPORT
pub fn rasterize_objects(
    img: NdArray<u8>, geometry_lengths: NdArray<i32>, polygon_lengths: NdArray<i32>,
    polygons: NdArray<f64>, label: u8, num_threads: i32) {
        
    assert!(num_threads > 0);
        
    let height = img.shape[0] as f64;
    let width = img.shape[1] as f64;
    
    let height_factor = 1. / height;
    let width_factor = 1. / width;
    
    let half_pel_y = 0.5 * height_factor;
    let half_pel_x = 0.5 * width_factor;
        
    let img = Arc::new(img);
    let mut threads = Vec::new();

    for i in 0..geometry_lengths.shape[0] {
        let mut left = f64::INFINITY;
        let mut right = -f64::INFINITY;
        let mut top = f64::INFINITY;
        let mut bottom = -f64::INFINITY;
        let mut geometry = Vec::new();
        
        for j in 0..geometry_lengths[i] {
            let k = polygon_lengths[[i, j]];
            let polygon = polygons.view(0, i).view(0, j).view_range(0, 0, k);
            
            for l in 0..polygon.shape[0] {
                let x = polygon[[l, 0]];
                left = left.min(x);
                right = right.max(x);
                
                let y = polygon[[l, 1]];                
                top = top.min(y);
                bottom = bottom.max(y);
            }
            
            let clockwise = is_clockwise(&polygon);
            geometry.push((polygon, clockwise));
        }
                    
        let left = (img.shape[1] as i32).min((left * width) as i32).max(0);
        let right = (img.shape[1] as i32).min(1 + (right * width) as i32).max(0);
        let top = (img.shape[0] as i32).min((top * height) as i32).max(0);
        let bottom = (img.shape[0] as i32).min(1 + (bottom * height) as i32).max(0);
            
        let left = Arc::new(left);
        let right = Arc::new(right);
        let top = Arc::new(top);
        let bottom = Arc::new(bottom);
        let geometry = Arc::new(geometry);
            
        for thread_id in 0..num_threads {
            let img = img.clone();
            let left = left.clone();
            let right = right.clone();
            let top = top.clone();
            let bottom = bottom.clone();
            let geometry = geometry.clone();
            
            let handle = std::thread::spawn(move || {
                let mut img = (*img).clone();
                    
                for y in *top..*bottom {
                    if y % num_threads != thread_id {
                        continue;
                    }
                        
                    let fy = ((y as f64) * height_factor) + half_pel_y;
                        
                    for x in *left..*right {
                        let fx = ((x as f64) * width_factor) + half_pel_x;
                    
                        let mut inside_object = false;
                        let mut inside_hole = false;
                        
                        for (polygon, clockwise) in geometry.iter()  {
                            if is_inside(polygon, fx, fy) {
                                if *clockwise {
                                    inside_object = true;
                                } else {
                                    inside_hole = true;
                                }
                            }
                        }

                        if inside_object && !inside_hole {
                            img[[y, x]] = label;
                        }
                    }
                }
            });
            
            threads.push(handle);
        }
            
        while !threads.is_empty() {
            threads.pop().unwrap().join().unwrap();
        }
    }
}

fn fill(img: &mut NdArray<u8>, cy: i32, cx: i32, votes: &mut NdArray<usize>, ignored_votes: usize) -> bool {
    for i in 0..votes.shape[0] {
        votes[i] = 0;
    }
    
    for y in (cy-1).max(0)..(cy+2).min(img.shape[0]) {
        for x in (cx-1).max(0)..(cx+2).min(img.shape[1]) {
            let mut val = img[[y, x]];
            if val == 255 {
                val = 15;
            }
            votes[val] += 1;
        }
    }
    votes[15] -= ignored_votes;
    
    let mut majority = 0;
    for i in 1..votes.shape[0] {
        if votes[i] > votes[majority] {
            majority = i;
        }
    }
    
    if majority < 15 {
        img[[cy, cx]] = majority as u8;
        return true;
    }
    
    false
}

//PYTHON_EXPORT
pub fn flood_fill_holes(mut img: NdArray<u8>) {
    let mut undefined_pixels = Vec::new();
    for y in 0..img.shape[0] {
        for x in 0..img.shape[1] {
            if img[[y, x]] == 255 {
                undefined_pixels.push((y, x));
            }
        }
    }
    
    ndarray!(votes -> 0usize; 16);
    let mut ignored_votes = 1; // don't include the center pixel
    let mut changed_pixels = Vec::new();
    
    while !undefined_pixels.is_empty() {
        changed_pixels.clear();
        for i in 0..undefined_pixels.len() {
            let (y, x) = undefined_pixels[i];
            let result = fill(&mut img, y, x, &mut votes, ignored_votes);
            if result {
                changed_pixels.push(i);
            }
        }
        
        if changed_pixels.is_empty() {
            ignored_votes += 1; // ignore an additional 'undefined' vote to eventually grow defined region again
        } else {
            ignored_votes = 1;
            changed_pixels.reverse();
            for i in &changed_pixels {
                undefined_pixels.remove(*i);
            }
        }
    }
}
