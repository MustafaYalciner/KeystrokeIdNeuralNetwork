import csv


class DataTrigraph:

    if __name__ == '__main__':
        PREV_KEY_INDEX = 2
        USER_ID_INDEX = 16
        print('Test')
        all = []
        input_file = open('data/featureset.csv')
        with open('data/output.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(input_file)
            header = next(reader)
            header.append('KEYCODE_TRI')
            writer.writerow(header)
            print(header)
            previousRow = next(reader)
            prevKeycodeInPrevRow = previousRow[PREV_KEY_INDEX]
            userInPrevRow = previousRow[USER_ID_INDEX]
            previousRow.append(previousRow[PREV_KEY_INDEX])
            writer.writerow(previousRow)
            print('keycode', prevKeycodeInPrevRow)
            print('user', userInPrevRow)
            for item in reader:
                if item[USER_ID_INDEX] == userInPrevRow:
                    item.append(prevKeycodeInPrevRow)
                else:
                    item.append(item[PREV_KEY_INDEX])
                prevKeycodeInPrevRow = item[PREV_KEY_INDEX]
                userInPrevRow = item[USER_ID_INDEX]
                writer.writerow(item)

            input_file.close()
            print('success')