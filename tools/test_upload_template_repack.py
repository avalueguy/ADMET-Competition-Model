"""Meaningful quoting-boundary checks for exact-byte CSV prediction insertion."""
import csv
import io
import unittest

from repack_upload_template import field_spans


class Boundaries(unittest.TestCase):
    def test_csv_field_boundaries(self):
        samples=[
            b'a,b,c,d,y,f\r\nx,0,1,"C,C",,MAE\r\n',
            b'a,b,c,d,y,f\nx,0,1,"C""C","",MAE\n',
            b'a,b,c,d,y,f\nx,0,1,"C\nC",,MAE',
            b'\xef\xbb\xbfa,b,c,d,y,f\r\nx,0,1,C,,MAE\r\n',
        ]
        for raw in samples:
            with self.subTest(raw=raw):
                spans=field_spans(raw)
                rows=list(csv.reader(io.StringIO(raw.decode("utf-8-sig"),newline="")))
                self.assertEqual(len(spans),len(rows))
                for row,fields in zip(rows,spans):
                    self.assertEqual(len(fields),6)
                    for expected,(lo,hi) in zip(row,fields):
                        parsed=next(csv.reader(io.StringIO(raw[lo:hi].decode()+",END",newline="")))[0]
                        self.assertEqual(expected,parsed)


if __name__=="__main__":unittest.main(verbosity=2)
