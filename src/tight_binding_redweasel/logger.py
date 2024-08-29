# This is a class that captures an optimisation log.
# This is useful for analysing the convergence of the methods.
# It is also very useful to control how the fitting process is displayed.

class OptimisationLogger:
    def __init__(self, print_loss=True, update_line=True, verbose=True):
        self.stream = []
        self.iterations = []
        self.loss = []
        self.verbose = verbose
        self.print_loss = print_loss
        self.update_line = update_line

    def add_message(self, msg):
        self.stream.append(msg)
        if self.verbose:
            if self.update_line:
                # don't put these messages into the update line
                print()
            print(msg)

    def add_data(self, iteration, loss, max_err):
        self.stream.append((iteration, loss, max_err))
        self.iterations.append(iteration)
        self.loss.append(loss)
        if self.print_loss:
            if self.update_line:
                print(end="\r")
            # error visualizer
            # U+2581 = ▁ ... U+2588 = █
            scale = max(*max_err)
            error_vis = "".join([" " if err/scale <= 1/16 else chr(0x2581 + round(err/scale * 8 - 1)) for err in max_err])
            print(f"{iteration:3}: loss{loss:9.2e}{scale:8.1e}×[{error_vis}]", end="")
            if not self.update_line:
                print()
    
    def iteration_count(self):
        return self.iterations[-1]
    
    def last_loss(self):
        return self.loss[-1]

    def plot_loss(self):
        from matplotlib import pyplot as plt
        plt.semilogy(self.iterations, self.loss)
    