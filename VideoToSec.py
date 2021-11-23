import cv2
from PlateFinderOCR import PLFinder

class Vid2Seq:
    vidcap = None
    PL = None
    plates = dict()
    def __init__(self,vid_path):
        self.vidcap = cv2.VideoCapture(vid_path)
        self.PL = PLFinder()

    def convert_all(self):
        success, image = self.vidcap.read()
        count = 0
        while success:

            success, image = self.vidcap.read()
            if not success:
                break
            self.PL.next_image(image)

            plate = self.PL.get_image()

            if plate.shape[0] != image.shape[0]:
                x = self.PL.OCR()
                if x != []:
                    if x[0][-2] not in self.plates:
                        self.plates[x[0][-2]]=1
                    else:
                        self.plates[x[0][-2]]+=1
                image[0:plate.shape[0], 0:plate.shape[1]] = plate


            cv2.imshow("Frame", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            print('count:', count)
            count += 1

    def convert_n(self,n):
        for i in range(n):
            success, image = self.vidcap.read()
            if not success:
                break
            self.PL.next_image(image)
            plate = self.PL.get_image()
            if plate.shape[0] != image.shape[0]:
                x = self.PL.OCR()
                print(x)
                if x!=[]:
                    self.plates.add(x[0][-2])
                image[0:plate.shape[0], 0:plate.shape[1]] = plate
            cv2.imshow("Frame",image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            print('count:', i)







#{'HsO ': 2, 'HSPTx': 2, 'IulGYBI': 1, 'IOIGYBI 1231': 1, 'Mic Yei': 1, 'MiGYel': 2, 'WGY9 121': 1, 'MGyej 121': 2, 'Tidoye 121': 1, 'TO1Ge 121': 1, 'TOlolb': 1, 'P959': 1, '2959': 3, '29591k': 1, 'P95941': 2, '29524k': 2, '0959': 1, 'K095 Tc 9': 1, '095': 2, 'K095 ic 9': 1, '6095ic 9': 1, 'K095 Tc9': 1, 'K095TC': 1, 'K095 TC 9': 1, 'K095 1C 9': 2, 'K095TC 9': 2, '0951c': 1, 'K09STC 93': 3, 'K095tc %': 1, '095TC': 2, '095TC 93': 4, '095TC 9': 3, '095TC 53': 1, '095TC %3': 1, '0095 TC 9': 1, '095Tc 9': 1, '095tc93': 1, '095Tc 93': 1, '(': 1, '[016/8 423': 1, '101b18 12': 1, '101b18 124': 1, 'llbte42Y': 1, '10 1678 122': 1, '[01678122': 1, '601618 23': 1, '601618 123': 2, 'WO16Y8 12': 1, '00164812': 1, '0016+8 124': 1, 'H016Y8 122': 1, 'I016Y8 122': 2, 'I016Y8123': 3, 'K120HK %': 1, '720HK 93': 2, '6,04': 2, '606772 4': 1, '606. 1,': 1, 'XGo6hT 129': 1, 'Go6ht 125': 2, 'Go6ht 12,': 1, 'Go641 12': 1, '(60641': 1, '60641 123': 2, '6064112': 1, 'BBZOHM': 1, "'B82OHMIIB": 1, "'B82OHM": 3, 'B82OHM': 2, "'B82OHMZI6": 1, 'B8ZOHM': 6, "'BBZOHM": 3, "'B8ZOHM": 1, '30y': 1}
#H591YY123 T016YB123 P959YK123
# K095TC93 T016YB123  X720HK93 Y606HT123
#B820HM716


if __name__ == "__main__":

    V2S = Vid2Seq("Untitled.mp4")
    V2S.convert_all()
    print(V2S.plates)
