import sys

def trace_calls(frame, event, arg):
    '''
    Taken from: https://pymotw.com/2/sys/tracing.html
    '''
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'write':
        # Ignore write() calls from print statements
        return
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    caller = frame.f_back
    caller_line_no = caller.f_lineno
    caller_filename = caller.f_code.co_filename
    print('Call to %s on line %s of %s from line %s of %s' % \
        (func_name, func_line_no, func_filename,
         caller_line_no, caller_filename))
    return

def trace_local_calls(path, exclude=[]):
    '''
    Returns a pimped trace function that can be used to only track function
    calls within a directory, which is specified using the path argument.

    You can also exclude certain functions from being reported (for instance
    functions that are iterated many times and thus clutter the output), by
    passing them to the exclude parameter.

    To use it, do:
    >>> import sys
    >>> trace_calls = trace_local_calls('my_path', exclude=['this_functions'])
    >>> sys.settrace(trace_calls)

    '''

    if isinstance(exclude, str):
        exclude = [exclude]

    n = len(path)
    def trace_calls(frame, event, arg):
        if event != 'call':
            return
        co = frame.f_code
        func_name = co.co_name
        if func_name in ['write'] + exclude:
            # Ignore write() calls from print statements
            # and functions passed with the exclude argument
            return
        func_line_no = frame.f_lineno
        func_filename = co.co_filename

        if not func_filename[0:n] == path:
            return
        caller = frame.f_back
        caller_line_no = caller.f_lineno
        caller_filename = caller.f_code.co_filename
        print('Call to %s on line %s of %s from line %s of %s' % \
              (func_name, func_line_no, func_filename,
               caller_line_no, caller_filename))
        return
    return trace_calls
