from poet_2.alphabets import Uniprot21


class Alphabet(Uniprot21):
    def __init__(self):
        chars = b"ARNDCQEGHILKMFPSTWYVOUBZX"
        mask_token = len(chars) - 1
        chars += b"-"
        gap_token = len(chars) - 1
        chars += b"$"
        start_token = len(chars) - 1
        chars += b"*"
        stop_token = len(chars) - 1
        chars += b"|"
        cls_token = len(chars) - 1

        super(Uniprot21, self).__init__(
            chars, encoding=None, mask=True, missing=mask_token
        )

        self.mask_token = mask_token
        self.gap_token = gap_token
        self.start_token = start_token
        self.stop_token = stop_token
        self.cls_token = cls_token


class S3DiAlphabet(Uniprot21):
    def __init__(self):
        # NOTE: padding this alphabet so that special tokens match between this alphabet
        #       and Alphabet above
        chars = b"ACDEFGHIKLMNPQRSTVWY    X"
        mask_token = len(chars) - 1
        chars += b"-"
        gap_token = len(chars) - 1
        chars += b"$"
        start_token = len(chars) - 1
        chars += b"*"
        stop_token = len(chars) - 1
        chars += b"|"
        cls_token = len(chars) - 1

        super(Uniprot21, self).__init__(
            chars, encoding=None, mask=True, missing=mask_token
        )

        self.mask_token = mask_token
        self.gap_token = gap_token
        self.start_token = start_token
        self.stop_token = stop_token
        self.cls_token = cls_token
