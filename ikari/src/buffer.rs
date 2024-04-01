use crate::renderer::USE_LABELS;

use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct GpuBuffer {
    src: wgpu::Buffer,
    capacity: usize,
    stride: usize,
    length: usize,
    usage: wgpu::BufferUsages,
}

impl GpuBuffer {
    pub fn empty(device: &wgpu::Device, stride: usize, usage: wgpu::BufferUsages) -> Self {
        let capacity = 1;
        Self {
            src: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: USE_LABELS.then_some("GpuBuffer"),
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

    #[profiling::function]
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
            panic!("Tried to create a buffer with data that wont fit in the capacity");
        }

        let mut contents_padded = initial_contents.to_vec();
        let padding_count = capacity_bytes - initial_contents.len();
        contents_padded.reserve(padding_count);
        contents_padded.resize(capacity_bytes, 0);

        Self {
            src: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: USE_LABELS.then_some("GpuBuffer"),
                contents: &contents_padded,
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

    /// returns true if the buffer was resized
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

    pub fn destroy(&self) {
        self.src.destroy();
    }
}

#[derive(Debug)]
pub struct ChunkedBuffer<T, ID> {
    buffer: Vec<u8>,
    chunks: Vec<BufferChunk<ID>>,
    biggest_chunk_length: usize,
    item_type: std::marker::PhantomData<T>,
}

#[derive(Debug)]
pub struct BufferChunk<ID> {
    pub id: ID,
    pub start_index: usize,
    pub end_index: usize,
}

impl<T: bytemuck::Pod, ID> ChunkedBuffer<T, ID> {
    pub fn new() -> Self {
        Self {
            buffer: vec![],
            chunks: vec![],
            biggest_chunk_length: 0,
            item_type: std::marker::PhantomData,
        }
    }

    /// Takes an iterator of chunks (Vec<T>) and places them all into a buffer, keeping track of the byte ranges of each chunk.
    /// The chunks are aligned by `alignment`.
    /// Extra space is added to the end of the buffer to avoid the 'Dynamic binding at index x with offset y would overrun the buffer' error.
    /// Only supports replace() for now, could maybe create an add_all function if we keep track of the biggest chunk length and fix the padding
    /// at the end of the buffer
    #[profiling::function]
    pub fn replace(&mut self, chunks: impl Iterator<Item = (ID, Box<[T]>)>, alignment: usize) {
        self.clear();

        let mut wasted_bytes = 0;
        let mut total_bytes = 0;

        let stride = std::mem::size_of::<T>();

        for (id, chunk) in chunks {
            let start_index = self.buffer.len();
            let end_index = start_index + chunk.len() * stride;
            let chunk_bytes = bytemuck::cast_slice(&chunk);
            let chunk_length = chunk_bytes.len();

            self.buffer.extend_from_slice(chunk_bytes);

            // add padding
            let needed_padding = alignment - (self.buffer.len() % alignment);

            self.buffer.resize(self.buffer.len() + needed_padding, 0);

            if chunk.len() > self.biggest_chunk_length {
                self.biggest_chunk_length = chunk.len();
            }

            self.chunks.push(BufferChunk {
                id,
                start_index,
                end_index,
            });

            if log::log_enabled!(log::Level::Debug) {
                total_bytes += needed_padding + chunk_length;
                wasted_bytes += needed_padding;
            }
        }

        if total_bytes > 0 {
            log::debug!(
                "wasted_bytes={wasted_bytes}, total_bytes={total_bytes} ({:.4}%)",
                100.0 * wasted_bytes as f32 / total_bytes as f32
            );
        }

        // to avoid 'Dynamic binding at index x with offset y would overrun the buffer' error
        self.buffer
            .resize(self.buffer.len() + self.biggest_chunk_length * stride, 0);
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.chunks.clear();
        self.biggest_chunk_length = 0;
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    pub fn chunks(&self) -> &[BufferChunk<ID>] {
        &self.chunks
    }

    pub fn stride(&self) -> usize {
        std::mem::size_of::<T>()
    }

    pub fn biggest_chunk_length(&self) -> usize {
        self.biggest_chunk_length
    }
}

impl<T: bytemuck::Pod, ID> Default for ChunkedBuffer<T, ID> {
    fn default() -> Self {
        Self::new()
    }
}
