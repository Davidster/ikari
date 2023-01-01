use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct GpuBuffer {
    src: wgpu::Buffer,
    capacity: usize, // capacity in bytes = capacity * stride
    stride: usize,
    length: usize, // length in bytes = length * stride
    usage: wgpu::BufferUsages,
}

impl GpuBuffer {
    pub fn empty(device: &wgpu::Device, stride: usize, usage: wgpu::BufferUsages) -> Self {
        let capacity = 1;
        Self {
            src: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuBuffer"),
                contents: &vec![0u8; capacity * stride],
                usage,
            }),
            capacity,
            stride,
            length: 0,
            usage,
        }
    }

    pub fn from_bytes(
        device: &wgpu::Device,
        initial_contents: &[u8],
        stride: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let capacity = (initial_contents.len() as f32 / stride as f32).ceil() as usize;
        Self::from_bytes_and_capacity(device, initial_contents, stride, capacity, usage)
    }

    pub fn from_bytes_and_capacity(
        device: &wgpu::Device,
        initial_contents: &[u8],
        stride: usize,
        capacity: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let capacity_bytes = capacity * stride;
        let length = (initial_contents.len() as f32 / stride as f32).ceil() as usize;

        if length > capacity {
            panic!("Tried to create a buffer with data that won't fit in the capacity");
        }

        let padding: Vec<u8> = (0..(capacity_bytes - initial_contents.len()))
            .map(|_| 0u8)
            .collect();

        let contents: Vec<u8> = initial_contents
            .iter()
            .chain(padding.iter())
            .copied()
            .collect();

        Self {
            src: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuBuffer"),
                contents: &contents,
                usage,
            }),
            capacity,
            stride,
            length,
            usage,
        }
    }

    pub fn src(&self) -> &wgpu::Buffer {
        &self.src
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn length_bytes(&self) -> usize {
        self.length * self.stride
    }

    pub fn _capacity(&self) -> usize {
        self.capacity
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity * self.stride
    }

    // returns true if the buffer was resized
    pub fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        let new_length = (data.len() as f32 / self.stride as f32).ceil() as usize;

        if new_length <= self.capacity {
            queue.write_buffer(&self.src, 0, data);
            self.length = new_length;
            false
        } else {
            // make a new buffer with 2x the size
            self.src.destroy();
            let new_buffer_capacity = new_length * 2;
            *self = Self::from_bytes_and_capacity(
                device,
                data,
                self.stride,
                new_buffer_capacity,
                self.usage,
            );
            true
        }
    }
}

#[derive(Debug)]
pub struct ChunkedBuffer<T> {
    buffer: Vec<u8>,
    chunks: Vec<BufferChunk>,
    biggest_chunk_length: usize,
    item_type: std::marker::PhantomData<T>,
}

#[derive(Debug)]
pub struct BufferChunk {
    pub id: usize,
    pub start_index: usize,
    pub end_index: usize,
}

impl<T: bytemuck::Pod> ChunkedBuffer<T> {
    pub fn empty() -> Self {
        Self {
            buffer: vec![],
            chunks: vec![],
            biggest_chunk_length: 0,
            item_type: std::marker::PhantomData,
        }
    }
    /*
        Takes an iterator of chunks (Vec<T>) and places them all into a buffer, keeping track of the byte ranges of each chunk.
        The chunks are aligned by `alignment`.
        Extra space is added to the end of the buffer to avoid the 'Dynamic binding at index x with offset y would overrun the buffer' error.
    */
    pub fn new(chunks: impl Iterator<Item = (usize, Vec<T>)>, alignment: usize) -> Self {
        let mut biggest_chunk_length = 0;
        let mut buffer: Vec<u8> = vec![];
        let mut buffer_chunks: Vec<BufferChunk> = vec![];
        let stride = std::mem::size_of::<T>();

        for (id, chunk) in chunks {
            let start_index = buffer.len();
            let end_index = start_index + chunk.len() * stride;
            buffer.append(&mut bytemuck::cast_slice(&chunk).to_vec());

            // add padding
            let needed_padding = alignment - (buffer.len() % alignment);
            let mut padding: Vec<_> = (0..needed_padding).map(|_| 0u8).collect();
            buffer.append(&mut padding);

            if chunk.len() > biggest_chunk_length {
                biggest_chunk_length = chunk.len();
            }

            buffer_chunks.push(BufferChunk {
                id,
                start_index,
                end_index,
            })
        }

        // to avoid 'Dynamic binding at index x with offset y would overrun the buffer' error
        let mut max_instances_padding: Vec<_> =
            (0..(biggest_chunk_length * stride)).map(|_| 0u8).collect();
        buffer.append(&mut max_instances_padding);

        Self {
            buffer,
            chunks: buffer_chunks,
            biggest_chunk_length,
            item_type: std::marker::PhantomData,
        }
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    pub fn chunks(&self) -> &[BufferChunk] {
        &self.chunks
    }

    pub fn stride(&self) -> usize {
        std::mem::size_of::<T>()
    }

    pub fn biggest_chunk_length(&self) -> usize {
        self.biggest_chunk_length
    }
}
