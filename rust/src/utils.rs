pub struct NdArray<T> {
    data: *const T,
    pub shape: Vec<i32>,
    strides: Vec<isize>
}

unsafe impl<T> Send for NdArray<T> {}
unsafe impl<T> Sync for NdArray<T> {}

impl<T> NdArray<T> where T: Copy {
    pub fn new(buffer: &mut Vec<T>, initial: T, shape: Vec<i32>) -> NdArray<T> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len()-1).rev() {
            strides[i] = strides[i+1] * (shape[i+1] as isize);
        }
        buffer.resize((strides[0] as usize) * (shape[0] as usize), initial);
        
        NdArray {
            data: buffer.as_ptr(),
            shape: shape.clone(),
            strides: strides
        }
    }
    
    // use this function very carefully as it can be abused to break Rust's safety guarantees!
    pub fn from_ptr(ptr: *const T, shape: *const i32, strides_in: *const i32, num_dims: i32) -> NdArray<T> {
        let num_dims = num_dims as usize;
        let mut strides = Vec::with_capacity(num_dims);
        let item_size = std::mem::size_of::<T>() as isize;
        
        unsafe {
            let shape = std::slice::from_raw_parts(shape, num_dims).to_vec();
            for stride in std::slice::from_raw_parts(strides_in, num_dims) {
                strides.push((*stride as isize) / item_size);
            }
            
            NdArray {
                data: ptr,
                shape: shape.clone(),
                strides: strides
            }
        }
    }
    
    // use this function very carefully as it can be abused to break Rust's safety guarantees!
    pub fn clone(&self) -> NdArray<T> {
        NdArray {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone()
        }
    }
    
    // use this function very carefully as it can be abused to break Rust's safety guarantees!
    pub fn view(&self, dim: usize, i: i32) -> NdArray<T> {
        debug_assert!(dim < self.shape.len());
        debug_assert!(i < self.shape[dim]);
        
        let mut shape = self.shape.clone();
        shape.remove(dim);
        let mut strides = self.strides.clone();
        strides.remove(dim);
        
        unsafe {
            NdArray {
                data: self.data.offset((i as isize) * self.strides[dim]),
                shape: shape,
                strides: strides
            }
        }
    }
    
    // use this function very carefully as it can be abused to break Rust's safety guarantees!
    pub fn view_range(&self, dim: usize, low: i32, high: i32) -> NdArray<T> {
        debug_assert!(dim < self.shape.len());
        debug_assert!(low < high);
        debug_assert!(0 <= low);
        debug_assert!(high <= self.shape[dim]);
        
        let mut shape = self.shape.clone();
        shape[dim] = high - low;
        
        unsafe {
            NdArray {
                data: self.data.offset((low as isize) * self.strides[dim]),
                shape: shape,
                strides: self.strides.clone()
            }
        }
    }
}

macro_rules! generate_index_ops {
    ( compute_index: $s: ident, $i: ident, $n: literal -> $result: ident ) => {
        let mut $result = 0;
        for i in 0..$n {
            $result += ($i[i] as isize) * $s.strides[i];
        }
    };
    
    ( $n: literal, $t: ty ) => {
        impl<T> std::ops::Index<[$t; $n]> for NdArray<T> {
            type Output = T;
    
            fn index(&self, i: [$t; $n]) -> &Self::Output {
                debug_assert_eq!(self.shape.len(), $n);
                generate_index_ops!(compute_index: self, i, $n -> result);
                unsafe {
                    &*self.data.offset(result)
                }
            }
        }
        
        impl<T> std::ops::IndexMut<[$t; $n]> for NdArray<T> {
            fn index_mut(&mut self, i: [$t; $n]) -> &mut Self::Output {
                debug_assert_eq!(self.shape.len(), $n);
                generate_index_ops!(compute_index: self, i, $n -> result);
                unsafe {
                    &mut *(self.data.offset(result) as *mut T)
                }
            }
        }
    };
    
    ( $t: ty ) => {
        impl<T> std::ops::Index<$t> for NdArray<T> {
            type Output = T;
    
            fn index(&self, i: $t) -> &Self::Output {
                debug_assert_eq!(self.shape.len(), 1);
                unsafe {
                    &*self.data.offset((i as isize) * self.strides[0])
                }
            }
        }
        
        impl<T> std::ops::IndexMut<$t> for NdArray<T> {
            fn index_mut(&mut self, i: $t) -> &mut Self::Output {
                debug_assert_eq!(self.shape.len(), 1);
                unsafe {
                    &mut *(self.data.offset((i as isize) * self.strides[0]) as *mut T)
                }
            }
        }
        
        generate_index_ops!(1, $t);
        generate_index_ops!(2, $t);
        generate_index_ops!(3, $t);
        generate_index_ops!(4, $t);
        generate_index_ops!(5, $t);
        generate_index_ops!(6, $t);
        generate_index_ops!(7, $t);
        generate_index_ops!(8, $t);
        generate_index_ops!(9, $t);
    }
}

generate_index_ops!(i8);
generate_index_ops!(i16);
generate_index_ops!(i32);
generate_index_ops!(i64);
generate_index_ops!(isize);
generate_index_ops!(u8);
generate_index_ops!(u16);
generate_index_ops!(u32);
generate_index_ops!(u64);
generate_index_ops!(usize);

macro_rules! ndarray {
    ( $name: ident -> $initial: expr; $($shape: expr),+ ) => {
        let mut $name = vec![$initial];
        let mut $name = NdArray::new(&mut $name, $initial, vec![ $($shape),+ ] );
    };
}
