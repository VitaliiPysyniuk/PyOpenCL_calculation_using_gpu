import time
import numpy as np
import pyopencl as OpenCL

MIN_VALUE = -100
MAX_VALUE = 100

kernel_src_1 = '''
__kernel void adding(__global double* a, 		      
	                     __global double* b,
				         __global double* c) {	
	int i = get_global_id(0);
	c[i] = a[i] + b[i];
}
'''

kernel_src_2 = '''
__kernel void subtracting(__global double* a, 		      
	                          __global double* b,
				              __global double* c) {	
	int i = get_global_id(0);
	c[i] = a[i] - b[i];
}
'''

kernel_src_3 = '''
__kernel void multiplication(int w_a, int w_b,
                              __global double* a,
                              __global double* b,
                              __global double* c){
    int i = get_global_id(0);
    int a_row = i / w_b;
    int b_col = i % w_b;
    double value = 0.0f;
    for (int k = 0; k < w_a; k++)
        value += a[a_row * w_a + k] * b[k * w_b + b_col];
    c[i] = value;
}
'''

kernel_src_4 = '''
kernel void multiplication_scalar(__global double* a,
                            __global double* b,
                            __global double* c) {
    int i = get_global_id(0);
    c[i] = a[i] * b[0];
}
'''


def print_matrix(matrix):
    print(f'Розмірність результуючої матриці: {matrix.shape[0]}x{matrix.shape[1]}')
    for row in matrix:
        print('[ ', end='')
        for item in row:
            print(item, end=' ')
        print(']')
    print('\n')


def create_host_buffer(hostbuf):
    buffer = OpenCL.Buffer(context=context, flags=OpenCL.mem_flags.READ_ONLY | OpenCL.mem_flags.COPY_HOST_PTR,
                           hostbuf=hostbuf)
    return buffer


def create_device_buffer(size):
    buffer = OpenCL.Buffer(context=context, flags=OpenCL.mem_flags.READ_WRITE, size=size)
    return buffer


platform = OpenCL.get_platforms()[0]
devices = platform.get_devices(OpenCL.device_type.GPU)
nvidia_gpu = devices[0]

context = OpenCL.Context(devices)

queue = OpenCL.CommandQueue(context, nvidia_gpu, OpenCL.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

program_src_1 = OpenCL.Program(context, kernel_src_1).build(devices=[nvidia_gpu])
program_src_2 = OpenCL.Program(context, kernel_src_2).build(devices=[nvidia_gpu])
program_src_3 = OpenCL.Program(context, kernel_src_3).build(devices=[nvidia_gpu])
program_src_4 = OpenCL.Program(context, kernel_src_4).build(devices=[nvidia_gpu])


while True:
    matrix_dimension = int(input('Введіть розмірність матриць: '))
    show_results = True if input('Виводити проміжні результи (y/n): ') == 'y' else False
    start_time = time.time()

    C2 = []
    for i in range(matrix_dimension):
        C2.append([21 * (i**2 - 2 * j) for j in range(matrix_dimension)])
    C2 = np.array(object=C2, dtype=np.double)
    C2_h, C2_w = C2.shape
    C2_buf = create_host_buffer(C2)

    b2 = []
    for i in range(1, matrix_dimension + 1):
        b2.append([21 / i**4])
    b2 = np.array(object=b2, dtype=np.double)
    b2_h, b2_w = b2.shape
    b2_buf = create_host_buffer(b2)

    A2 = ((MAX_VALUE - MIN_VALUE) * np.random.random((matrix_dimension, matrix_dimension)) + MIN_VALUE).astype('double')
    A2_h, A2_w = A2.shape
    A2_buf = create_host_buffer(A2)
    B2 = ((MAX_VALUE - MIN_VALUE) * np.random.random((matrix_dimension, matrix_dimension)) + MIN_VALUE).astype('double')
    B2_h, B2_w = B2.shape
    B2_buf = create_host_buffer(B2)
    b1 = (((MAX_VALUE - MIN_VALUE) * np.random.random((matrix_dimension, 1))) + MIN_VALUE).astype('double')
    b1_h, b1_w = b1.shape
    b1_buf = create_host_buffer(b1)
    c1 = (20 * (MAX_VALUE - MIN_VALUE) * np.random.random((matrix_dimension, 1)) + MIN_VALUE).astype('double')
    c1_h, c1_w = c1.shape
    c1_buf = create_host_buffer(c1)
    A1 = ((MAX_VALUE - MIN_VALUE) * np.random.random((matrix_dimension, matrix_dimension)) + MIN_VALUE).astype('double')
    A1_h, A1_w = A1.shape
    A1_buf = create_host_buffer(A1)
    A = ((MAX_VALUE - MIN_VALUE) * np.random.random((matrix_dimension, matrix_dimension)) + MIN_VALUE).astype('double')
    A_h, A_w = A.shape
    A_buf = create_host_buffer(A)

    initialization_finish_time = time.time()
    initialization_time = initialization_finish_time - start_time

    level_2 = {}
    level_2['val_1'] = np.empty((B2_h, A2_w), dtype=np.double)
    level_2['buf_1'] = create_device_buffer(B2.nbytes)
    level_2['val_2'] = np.empty((b1_h, c1_w), dtype=np.double)
    level_2['buf_2'] = create_device_buffer(b1.nbytes)
    program_src_3.multiplication(queue, (B2_h * A2_w,), None, np.int32(B2_w), np.int32(A2_w), B2_buf, A2_buf,
                                    level_2['buf_1'])
    program_src_1.adding(queue, (b1_h * c1_w,), None, b1_buf, c1_buf, level_2['buf_2'])
    OpenCL.enqueue_copy(queue, level_2['val_1'], level_2['buf_1'])
    OpenCL.enqueue_copy(queue, level_2['val_2'], level_2['buf_2'])

    if show_results:
        print('------------------------------- Результати виконання на 2 рівні -------------------------------')
        print_matrix(level_2['val_1'])
        print_matrix(level_2['val_2'])

    level_3 = {}
    level_3['val_1'] = np.empty((level_2['val_1'].shape[0], C2_w), dtype=np.double)
    level_3['buf_1'] = create_device_buffer(C2.nbytes)
    level_3['val_2'] = np.empty((A1_h, level_2['val_2'].shape[1]), dtype=np.double)
    level_3['buf_2'] = create_device_buffer(level_2['val_2'].nbytes)
    level_3['val_3'] = np.empty((A_h, b2_w), dtype=np.double)
    level_3['buf_3'] = create_device_buffer(b2.nbytes)
    program_src_2.subtracting(queue, (level_2['val_1'].shape[0] * C2_w,), None, level_2['buf_1'], C2_buf, level_3['buf_1'])
    program_src_3.multiplication(queue, (A1_h * level_2['val_2'].shape[1],), None, np.int32(A1_w), np.int32(level_2['val_2'].shape[1]),
                                    A1_buf, level_2['buf_2'], level_3['buf_2'])
    program_src_3.multiplication(queue, (A_h * b2_w,), None, np.int32(A_w), np.int32(b2_w), A_buf, b2_buf, level_3['buf_3'])
    OpenCL.enqueue_copy(queue, level_3['val_1'], level_3['buf_1'])
    OpenCL.enqueue_copy(queue, level_3['val_2'], level_3['buf_2'])
    OpenCL.enqueue_copy(queue, level_3['val_3'], level_3['buf_3'])
    level_3['t_val_2'] = level_3['val_2'].transpose()
    level_3['t_buf_2'] = create_host_buffer(level_3['t_val_2'])

    if show_results:
        print('------------------------------- Результати виконання на 3 рівні -------------------------------')
        print_matrix(level_3['val_1'])
        print_matrix(level_3['val_2'])
        print_matrix(level_3['val_3'])

    level_4 = {}
    level_4['val_1'] = np.empty((level_3['val_1'].shape[0], level_3['val_1'].shape[1]), dtype=np.double)
    level_4['buf_1'] = create_device_buffer(level_3['val_1'].nbytes)
    level_4['val_2'] = np.empty((level_3['t_val_2'].shape[0], level_3['val_2'].shape[1]), dtype=np.double)
    level_4['buf_2'] = create_device_buffer(np.dtype(np.double).itemsize)
    level_4['val_3'] = np.empty((level_3['val_1'].shape[0], level_3['val_3'].shape[1]), dtype=np.double)
    level_4['buf_3'] = create_device_buffer(level_3['val_3'].nbytes)
    program_src_3.multiplication(queue, (level_3['val_1'].size,), None, np.int32(level_3['val_1'].shape[1]), np.int32(level_3['val_1'].shape[1]),
                                    level_3['buf_1'], level_3['buf_1'], level_4['buf_1'])
    program_src_3.multiplication(queue, (level_3['t_val_2'].shape[0] * level_3['val_2'].shape[1],), None, np.int32(level_3['t_val_2'].shape[1]),
                                    np.int32(level_3['val_2'].shape[1]), level_3['t_buf_2'], level_3['buf_2'], level_4['buf_2'])
    program_src_3.multiplication(queue, (level_3['val_1'].shape[0] * level_3['val_3'].shape[1],), None, np.int32(level_3['val_1'].shape[1]),
                                    np.int32(level_3['val_3'].shape[1]), level_3['buf_1'], level_3['buf_3'], level_4['buf_3'])
    OpenCL.enqueue_copy(queue, level_4['val_1'], level_4['buf_1'])
    OpenCL.enqueue_copy(queue, level_4['val_2'], level_4['buf_2'])
    OpenCL.enqueue_copy(queue, level_4['val_3'], level_4['buf_3'])

    if show_results:
        print('------------------------------- Результати виконання на 4 рівні -------------------------------')
        print_matrix(level_4['val_1'])
        print_matrix(level_4['val_2'])
        print_matrix(level_4['val_3'])

    level_5 = {}
    level_5['val_1'] = np.empty((level_4['val_1'].shape[0], level_3['val_2'].shape[1]), dtype=np.double)
    level_5['buf_1'] = create_device_buffer(level_3['val_2'].nbytes)
    level_5['val_2'] = np.empty(level_4['val_3'].shape, dtype=np.double)
    level_5['buf_2'] = create_device_buffer(level_4['val_3'].nbytes)
    program_src_3.multiplication(queue, (level_4['val_1'].shape[0] * level_3['val_1'].shape[1],), None,
                                    np.int32(level_4['val_1'].shape[1]), np.int32(level_3['val_1'].shape[1]), level_4['buf_1'],
                                    level_3['buf_1'], level_5['buf_1'])
    program_src_4.multiplication_scalar(queue, (level_4['val_3'].size,), None, level_4['buf_3'], level_4['buf_2'], level_5['buf_2'])
    OpenCL.enqueue_copy(queue, level_5['val_1'], level_5['buf_1'])
    OpenCL.enqueue_copy(queue, level_5['val_2'], level_5['buf_2'])

    if show_results:
        print('------------------------------- Результати виконання на 5 рівні -------------------------------')
        print_matrix(level_5['val_1'])
        print_matrix(level_5['val_2'])

    level_6 = {}
    level_6['val'] = np.empty((level_5['val_1'].shape[0], level_5['val_2'].shape[1]), dtype=np.double)
    level_6['buf'] = create_device_buffer(level_5['val_1'].nbytes)
    program_src_3.multiplication(queue, (level_5['val_1'].size,), None, np.int32(level_5['val_1'].shape[1]),
                                    np.int32(level_5['val_2'].shape[1]), level_5['buf_1'], level_5['buf_2'], level_6['buf'])
    OpenCL.enqueue_copy(queue, level_6['val'], level_6['buf'])

    if show_results:
        print('------------------------------- Результати виконання на 6 рівні -------------------------------')
        print_matrix(level_6['val'])

    level_7 = {}
    level_7['val'] = np.empty((level_6['val'].shape[0], level_4['val_3'].shape[1]), dtype=np.double)
    level_7['buf'] = create_device_buffer(level_6['val'].nbytes)
    program_src_1.adding(queue, (level_6['val'].size,), None, level_6['buf'], level_4['buf_3'], level_7['buf'])
    OpenCL.enqueue_copy(queue, level_7['val'], level_7['buf'])
    level_7['t_val'] = level_7['val'].transpose()
    level_7['t_buf'] = create_host_buffer(level_7['t_val'])

    if show_results:
        print('------------------------------- Результати виконання на 7 рівні -------------------------------')
        print_matrix(level_7['val'])

    level_8 = {}
    level_8['val'] = np.empty((level_7['t_val'].shape[0], level_4['val_1'].shape[1]), dtype=np.double)
    level_8['buf'] = create_device_buffer(level_7['t_val'].nbytes)
    program_src_1.adding(queue, (level_7['t_val'].size,), None, level_7['t_buf'], level_4['buf_1'], level_8['buf'])
    OpenCL.enqueue_copy(queue, level_8['val'], level_8['buf'])


    calculation_finish_time = time.time()
    calculation_time = calculation_finish_time - initialization_finish_time
    all_time = calculation_finish_time - start_time

    print(f'\nЧас виконання: {all_time} с, з нього ініціалізація даних: {initialization_time} с, обчислення виразу: '
          f'{calculation_time} с')

    print('\nКінцевий результат обчислення у вигляді матриці-рядка: ')
    print_matrix(level_8['val'])




