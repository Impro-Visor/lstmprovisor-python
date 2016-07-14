from theano_lstm import LSTM
import theano
import theano.tensor as T
import util
import numpy as np

class PeekLSTM( LSTM ):
    def print_weights(self):
        print('In weights', self.in_gate.linear_matrix.get_value().shape, self.in_gate.linear_matrix.get_value())
        print('In biases', self.in_gate.bias_matrix.get_value().shape, self.in_gate.bias_matrix.get_value())
        print('Forget weights', self.forget_gate.linear_matrix.get_value().shape, self.forget_gate.linear_matrix.get_value())
        print('Forget biases', self.forget_gate.bias_matrix.get_value().shape, self.forget_gate.bias_matrix.get_value())
        print('Act weights', self.in_gate2.linear_matrix.get_value().shape, self.in_gate2.linear_matrix.get_value())
        print('Act biases', self.in_gate2.bias_matrix.get_value().shape, self.in_gate2.bias_matrix.get_value())
        print('Out weights', self.out_gate.linear_matrix.get_value().shape, self.out_gate.linear_matrix.get_value())
        print('Out biases', self.out_gate.bias_matrix.get_value().shape, self.out_gate.bias_matrix.get_value())

    def save_weights(self):
        np.save('tmp_save/in_gate_linear_matrix.npy',self.in_gate.linear_matrix.get_value())
        np.save('tmp_save/in_gate_bias_matrix.npy',self.in_gate.bias_matrix.get_value())
        np.save('tmp_save/forget_gate_linear_matrix.npy',self.forget_gate.linear_matrix.get_value())
        np.save('tmp_save/forget_gate_bias_matrix.npy',self.forget_gate.bias_matrix.get_value())
        np.save('tmp_save/in_gate2_linear_matrix.npy',self.in_gate2.linear_matrix.get_value())
        np.save('tmp_save/in_gate2_bias_matrix.npy',self.in_gate2.bias_matrix.get_value())
        np.save('tmp_save/out_gate_linear_matrix.npy',self.out_gate.linear_matrix.get_value())
        np.save('tmp_save/out_gate_bias_matrix.npy',self.out_gate.bias_matrix.get_value())

    def activate(self, x, h):
        """
        The hidden activation, h, of the network, along
        with the new values for the memory cells, c,
        Both are concatenated as follows:
        >      y = f( x, past )
        Or more visibly, with past = [prev_c, prev_h]
        > [c, h] = f( x, [prev_c, prev_h] )
        """

        if h.ndim > 1:
            #previous memory cell values
            prev_c = h[:, :self.hidden_size]

            #previous activations of the hidden layer
            prev_h = h[:, self.hidden_size:]
        else:

            #previous memory cell values
            prev_c = h[:self.hidden_size]

            #previous activations of the hidden layer
            prev_h = h[self.hidden_size:]

        x = theano.printing.Print("x",["shape", "__str__"])(x)
        x = util.Save("tmp_save/x")(x)
        prev_c = theano.printing.Print("prev_c",["shape", "__str__"])(prev_c)
        prev_c = util.Save("tmp_save/prev_c")(prev_c)
        prev_h = theano.printing.Print("prev_h",["shape", "__str__"])(prev_h)
        prev_h = util.Save("tmp_save/prev_h")(prev_h)

        # input and previous hidden constitute the actual
        # input to the LSTM:
        if h.ndim > 1:
            obs = T.concatenate([x, prev_h], axis=1)
        else:
            obs = T.concatenate([x, prev_h])
        obs = theano.printing.Print("obs",["shape", "__str__"])(obs)
        obs = util.Save("tmp_save/obs")(obs)
        # TODO could we combine these 4 linear transformations for efficiency? (e.g., http://arxiv.org/pdf/1410.4615.pdf, page 5)
        # how much to add to the memory cells
        in_gate = self.in_gate.activate(obs)
        in_gate = theano.printing.Print("in_gate",["shape", "__str__"])(in_gate)
        in_gate = util.Save("tmp_save/in_gate")(in_gate)

        # how much to forget the current contents of the memory
        forget_gate = self.forget_gate.activate(obs)
        forget_gate = theano.printing.Print("forget_gate",["shape", "__str__"])(forget_gate)
        forget_gate = util.Save("tmp_save/forget_gate")(forget_gate)

        # modulate the input for the memory cells
        in_gate2 = self.in_gate2.activate(obs)
        in_gate2 = theano.printing.Print("in_gate2",["shape", "__str__"])(in_gate2)
        in_gate2 = util.Save("tmp_save/in_gate2")(in_gate2)

        # new memory cells
        next_c = forget_gate * prev_c + in_gate2 * in_gate
        next_c = theano.printing.Print("next_c",["shape", "__str__"])(next_c)
        next_c = util.Save("tmp_save/next_c")(next_c)

        # modulate the memory cells to create the new output
        out_gate = self.out_gate.activate(obs)
        out_gate = theano.printing.Print("out_gate",["shape", "__str__"])(out_gate)
        out_gate = util.Save("tmp_save/out_gate")(out_gate)

        # new hidden output
        next_h = out_gate * T.tanh(next_c)
        next_h = theano.printing.Print("next_h",["shape", "__str__"])(next_h)
        next_h = util.Save("tmp_save/next_h")(next_h)
        next_h = T.opt.Assert("BLABHALDB")(next_h, T.all(next_h < next_h))

        if h.ndim > 1:
            return T.concatenate([next_c, next_h], axis=1)
        else:
            return T.concatenate([next_c, next_h])
