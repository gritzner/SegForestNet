pub struct NdArray<'a,T> {
    data: &'a [T],
    pub shape: Vec<i32>,
    strides: Vec<usize>
}

pub struct NdArrayMut<'a,T> {
    data: &'a mut [T],
    pub shape: Vec<i32>,
    strides: Vec<usize>
}

impl<'a,T> NdArray<'a,T> where T: Copy {
    fn get_metadata<U: Into<Vec<i32>>>(shape: U) -> (usize, Vec<i32>, Vec<usize>) {
        let shape = shape.into();
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len()-1).rev() {
            strides[i] = strides[i+1] * (shape[i+1] as usize);
        }
        
        (strides[0] * (shape[0] as usize), shape, strides)
    }
    
    pub fn new<U: Into<Vec<i32>>>(buffer: &'a mut Vec<T>, initial: T, shape: U) -> NdArrayMut<'a,T> {
        let metadata = NdArray::<T>::get_metadata(shape);
        buffer.resize(metadata.0, initial);
        
        unsafe {
            NdArrayMut {
                data: std::slice::from_raw_parts_mut(buffer.as_mut_ptr(), metadata.0),
                shape: metadata.1,
                strides: metadata.2
            }
        }
    }
    
    pub fn from_ptr<U: Into<Vec<i32>>>(ptr: *const T, shape: U) -> NdArray<'a,T> {
        let metadata = NdArray::<T>::get_metadata(shape);
        
        unsafe {
            NdArray {
                data: std::slice::from_raw_parts(ptr, metadata.0),
                shape: metadata.1,
                strides: metadata.2
            }
        }
    }
    
    pub fn from_ptr_mut<U: Into<Vec<i32>>>(ptr: *mut T, shape: U) -> NdArrayMut<'a,T> {
        let metadata = NdArray::<T>::get_metadata(shape);
        
        unsafe {
            NdArrayMut {
                data: std::slice::from_raw_parts_mut(ptr, metadata.0),
                shape: metadata.1,
                strides: metadata.2
            }
        }
    }
    
    pub fn create_view(&self, i: &[i32], low: i32, high: i32) -> NdArray<'a,T> {
        debug_assert!(i.len() < self.shape.len());
        debug_assert!(low < high);
        debug_assert!(0 <= low);
        debug_assert!(high <= self.shape[i.len()]);
        
        let mut offset = 0;
        for j in 0..i.len() {
            debug_assert!(0 <= i[j]);
            debug_assert!(i[j] < self.shape[j]);
            
            offset += (i[j] as usize) * self.strides[j];
        }
        offset += (low as usize) * self.strides[i.len()];

        let mut shape = vec![high - low];
        let mut strides = vec![self.strides[i.len()]];
        for j in i.len()+1..self.shape.len() {
            shape.push(self.shape[j]);
            strides.push(self.strides[j]);
        }
        
        let mut end = offset + 1;
        for j in 0..shape.len() {
            end += ((shape[j]-1) as usize) * strides[j];
        }
        
        NdArray {
            data: &self.data[offset..end],
            shape: shape,
            strides: strides
        }
    }
    
    // use this function very carefully as it can be abused to break Rust's safety guarantees!
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    // use this function very carefully as it can be abused to break Rust's safety guarantees!
    pub fn as_mut_ptr(&self) -> *mut T {
        self.data.as_ptr() as *mut T
    }
}

impl<'a,T> NdArrayMut<'a,T> where T: Copy {
    // use this function very carefully as it can be abused to break Rust's safety guarantees!
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    // use this function very carefully as it can be abused to break Rust's safety guarantees!
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl From<NdArray<'_,i32>> for Vec<i32> {
    fn from(ndarray: NdArray<'_,i32>) -> Self {
        ndarray.data.to_vec()
    }
}

impl From<&NdArray<'_,i32>> for Vec<i32> {
    fn from(ndarray: &NdArray<'_,i32>) -> Self {
        ndarray.data.to_vec()
    }
}

macro_rules! generate_index_ops {
    ( compute_index: $s: ident, $i: ident, $n: literal -> $result: ident ) => {
        let mut $result = $i[$n-1] as usize;
        for i in 0..$n-1 {
            $result += ($i[i] as usize) * $s.strides[i];
        }
    };
    
    ( $n: literal, $t: ty ) => {
        impl<T> std::ops::Index<[$t; $n]> for NdArray<'_,T> {
            type Output = T;
    
            fn index(&self, i: [$t; $n]) -> &Self::Output {
                debug_assert_eq!(self.shape.len(), $n);
                generate_index_ops!(compute_index: self, i, $n -> result);
                &self.data[result]
            }
        }
        
        impl<T> std::ops::Index<[$t; $n]> for NdArrayMut<'_,T> {
            type Output = T;
    
            fn index(&self, i: [$t; $n]) -> &Self::Output {
                debug_assert_eq!(self.shape.len(), $n);
                generate_index_ops!(compute_index: self, i, $n -> result);
                &self.data[result]
            }
        }
        
        impl<T> std::ops::IndexMut<[$t; $n]> for NdArrayMut<'_,T> {
            fn index_mut(&mut self, i: [$t; $n]) -> &mut Self::Output {
                debug_assert_eq!(self.shape.len(), $n);
                generate_index_ops!(compute_index: self, i, $n -> result);
                &mut self.data[result]
            }
        }
    };
    
    ( $t: ty ) => {
        impl<T> std::ops::Index<$t> for NdArray<'_,T> {
            type Output = T;
    
            fn index(&self, i: $t) -> &Self::Output {
                debug_assert_eq!(self.shape.len(), 1);
                &self.data[i as usize]
            }
        }
        
        impl<T> std::ops::Index<$t> for NdArrayMut<'_,T> {
            type Output = T;
    
            fn index(&self, i: $t) -> &Self::Output {
                debug_assert_eq!(self.shape.len(), 1);
                &self.data[i as usize]
            }
        }
        
        impl<T> std::ops::IndexMut<$t> for NdArrayMut<'_,T> {
            fn index_mut(&mut self, i: $t) -> &mut Self::Output {
                debug_assert_eq!(self.shape.len(), 1);
                &mut self.data[i as usize]
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
    ( new: $name: ident -> $initial: expr; [ $($shape: expr),+ ] ) => {
        let mut $name = vec![$initial];
        let mut $name = NdArray::new(&mut $name, $initial, vec![ $($shape),+ ] );
    };
    
    ( new: $name: ident -> $initial: expr; $shape: expr ) => {
        let mut $name = vec![$initial];
        let mut $name = NdArray::new(&mut $name, $initial, $shape );
    };
    
    ( from_ptr: $name: ident; [ $($shape: expr),+ ] ) => {
        let $name = NdArray::from_ptr($name, vec![ $($shape),+ ] );
    };
    
    ( from_ptr: $name: ident; $shape: expr ) => {
        let $name = NdArray::from_ptr($name, $shape );
    };
    
    ( from_ptr_mut: $name: ident; [ $($shape: expr),+ ] ) => {
        let mut $name = NdArray::from_ptr_mut($name, vec![ $($shape),+ ] );
    };
    
    ( from_ptr_mut: $name: ident; $shape: expr ) => {
        let mut $name = NdArray::from_ptr_mut($name, $shape );
    };
    
    // use this macro very carefully as it can be abused to break Rust's safety guarantees!
    ( from_ndarray: $name: ident ) => {
        let $name = NdArray::from_ptr($name.as_ptr(), $name.shape.clone());
    };

    // use this macro very carefully as it can be abused to break Rust's safety guarantees!
    ( from_ndarray_mut: $name: ident ) => {
        let mut $name = NdArray::from_ptr_mut($name.as_mut_ptr(), $name.shape.clone());
    };
}
