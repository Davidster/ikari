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
    pub fn empty(
        device: &wgpu::Device,
        capacity: usize, // TODO: make this optional?
        stride: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
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
