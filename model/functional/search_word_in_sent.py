import re

def perfectSearch(iword: str, sentence: str, get_bag_of_word = None):
    """
        Hàm kiểm tra xem một từ có xuất hiện trong một câu không

        :param iword: từ cần tìm
        :param sentence: câu mà hàm sẽ dùng để tìm từ đã cho
        :param get_bag_of_word: hàm để lấy những từ liên quan đến từ cần tìm (thường là từ đồng nghĩa)
        :return: boolean tof, int fix_length (tof == True nếu từ đó xuất hiện trong câu)

        :example:
            word = "ddrsram"
            sent = "this is ddr sram"
            tof, fix_length = perfectSearch(word, sent)
            tof: True
            fix_length: 2 (vì phải nối ddr và sram trong sent)
    """

    sent_len = len(sentence)

    matches = re.finditer(" ", sentence)
    start_positions = [0] + [match.start()+1 for match in matches] + [sent_len+1]

    if get_bag_of_word != None:
        bag_of_word = get_bag_of_word(iword)
    else:
        bag_of_word = [iword]

    for word in bag_of_word:
        for pos_idx in range(len(start_positions)):
            st_pos = start_positions[pos_idx]
            curr_pos_idx = pos_idx
            if (sent_len-st_pos) < len(word):
                break
                
            if (start_positions[pos_idx+1]-st_pos-1) <= len(word):
                for char in word:
                    if sentence[st_pos] == " ":
                        st_pos += 1
                        curr_pos_idx += 1
                    if char == sentence[st_pos]:
                        st_pos += 1
                    else:
                        break
                
                if st_pos == sent_len or st_pos == start_positions[curr_pos_idx+1]-1:
                    return True, (curr_pos_idx-pos_idx)
    return False, 0