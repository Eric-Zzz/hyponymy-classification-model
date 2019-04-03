import torch
from config import config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
_config = config()


def evaluate(golden_list, predict_list):
    pass


# def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#         if input.is_cuda:
#             igates = F.linear(input, w_ih)
#             hgates = F.linear(hidden[0], w_hh)
#             state = fusedBackend.LSTMFused.apply
#             return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
#
#         hx, cx = hidden
#         gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
#
#         ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#
#         ingate = torch.sigmoid(ingate)
#         forgetgate = torch.sigmoid(forgetgate)
#         cellgate = torch.tanh(cellgate)
#         outgate = torch.sigmoid(outgate)
#
#         cy = (forgetgate * cx) + (ingate * cellgate)
#         hy = outgate * torch.tanh(cy)
#
#         return hy, cy


    #lstm
def into_lstm(model,output,desorted_indices):
    output, state = model.char_lstm(output)
    output, _ = pad_packed_sequence(output, batch_first=True)
    output = output[desorted_indices]
    return output.view(output.shape[0], output.shape[1], 2, 50)

def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    #get the embed char
    input = model.char_embeds(batch_char_index_matrices)
    #reshape batch_word_len_lists to sort
    batch_word_len_lists=torch.reshape(batch_word_len_lists,[-1])
    #reshape to sort
    char_embeds=torch.reshape(input,[-1,input.shape[2],input.shape[3]])
    #padding
    index, sorted_lists = model.sort_input(batch_word_len_lists)
    _, unsorted_indices = torch.sort(index, descending=False)
    output_sequence = pack_padded_sequence(char_embeds[index], lengths=sorted_lists.data.tolist(),
                                           batch_first=True)

    out=into_lstm(model,output_sequence,unsorted_indices)

    # get forward and backword
    lstm_fw=torch.squeeze(torch.index_select(out,2,torch.LongTensor([0])))
    lstm_bw=torch.squeeze(torch.index_select(out,2,torch.LongTensor([1])))
    lstm_bw = torch.index_select(lstm_bw, 1, torch.LongTensor([0]))
    lstm_bw = torch.squeeze(lstm_bw).view(input.shape[0], input.shape[1], -1)
    #get the last char in words
    new=torch.ones(lstm_fw.shape[0],1,50)
    for i in range(batch_word_len_lists.shape[0]):
        new[i]=torch.index_select(lstm_fw[i],0,torch.LongTensor([batch_word_len_lists[i]-torch.LongTensor([1])]))
    lstm_fw=torch.squeeze(new)
    lstm_fw=lstm_fw.view(input.shape[0],input.shape[1],-1)

    return torch.cat([lstm_fw, lstm_bw], dim=-1)