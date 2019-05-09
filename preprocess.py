# -*- coding: utf-8 -*-
"""
@author: cqx931
2019
"""
import os
import json
import pygame
from pygame import ftfont
import pickle
import operator
import csv
import tsv

dict = open("data/VC/dict_fix.csv", "r")
edit = {}
cat = {}
data = {}
char2id = {}
id2char = {}
class2id = {}
id2class = {}
directory = "img_all/"

#counter
count = 0
dic_c = {}

def draw(c, idx, directory):
    # print(c,directory)
    pygame.init()
    font_t = ftfont.Font("fonts/NotoSansCJKtc-Regular.otf", 32)
    font_s = ftfont.Font("fonts/NotoSansCJKsc-Regular.otf", 32)
    rtext = font_t.render(c, True, (0, 0, 0), (255, 255, 255))
    if rtext is None:
        rtext = font_s.render(c, True, (0, 0, 0), (255, 255, 255))

    rtext = pygame.transform.scale(rtext, (32, 32))

    pygame.image.save(rtext, directory + str(idx) + ".jpg")

# create a clear csv file from the original dict file
def generateCSV():
    # read manual edit
    rawDict = open("data/VC/dictionary.txt", "r")

    with open('data/VC/edit.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            edit[rows[0]] = rows[1]

    with open('data/VC/dict.csv', 'w') as csvfile:
        fieldnames = ['character', 'decomposition','radical']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        for entry in rawDict:
            entry = json.loads(entry)
            c = entry["character"]
            d = entry["decomposition"]
            r = entry["radical"]
            if (c == r): d = 'X'

            if len(d) > 4 and c in edit:
                d = edit[c]
            if  len(d) == 4:
                if '⿳' in d:
                    d = '⿱' + d[1] + d[3]
                else:
                    print(c,d,r)

            writer.writerow({'character': c, 'decomposition': d, 'radical':r})

for idx,entry in enumerate(dict):
    count += 1

    entry = entry.split(',')
    if len(entry) <= 3:
        # from csv
        c = entry[0]
        d = entry[1]
        r = entry[2]
    else:
        # from raw json data
        entry = json.loads(entry)
        c = entry["character"]
        d = entry["decomposition"]
        r = entry["radical"]

    draw(c , idx, "data/VC/img_all/")

    r = r.strip("\n")
    id2char[idx] = c
    char2id[c] = idx
    patterns = []

    # unclassifiable common ch
    if c in "幻巧壮枣垂甚临览觉既释就疏聚舞疑敲黎蠢囊氓衅棘甥粤赫兢舔孵幾虧歸叢務發執鞏喬壯農報縣畝棗隸參鹹蠶斃鉆競營釋賴疊鹵釁粵":
        patterns.append("口")

    if len(d) == 3 and "X" not in d:
        ###### deal with traditional/simplified classes
        d = d.replace("鸟", "鳥")
        d = d.replace("马", "馬")
        d = d.replace("纟", "糹")
        d = d.replace("车", "車")
        d = d.replace("饣", "飠")
        d = d.replace("贝", "貝")
        d = d.replace("见", "見")
        d = d.replace("讠", "言")
        d = d.replace("钅", "釒")
        d = d.replace("车", "車")
        d = d.replace("门", "門")
        d = d.replace("风", "風")
        d = d.replace("白", "日")
        d = d.replace("区", "區")
        d = d.replace("页", "頁")
        d = d.replace("鱼", "魚")
        d = d.replace("齿", "齒")
        d = d.replace("韦", "韋")
        d = d.replace("麦", "麥")
        d = d.replace("尧", "堯")
        d = d.replace("爱", "愛")

        # d = d.replace("仓", "倉")
        # d = d.replace("乔", "喬")
        # d = d.replace("区", "區")
        # d = d.replace("夹", "夾")
        # d = d.replace("会", "會")
        # d = d.replace("争", "爭")
        # d = d.replace("韦", "韋")
        # 鹵
        ######
        # merge visually similar cases
        d = d.replace("鬥", "門")
        d = d.replace("⿸广", "⿸厂")
        d = d.replace("士", "土")
        d = d.replace("犬", "大")
        d = d.replace("良", "艮")
        d = d.replace("凡", "几")
        d = d.replace("氺", "水")
        d = d.replace("巴", "己")
        d = d.replace("巳", "己")
        d = d.replace("色", "己")
        d = d.replace("每", "母")
        d = d.replace("⿰手", "⿰扌")
        d = d.replace("島", "鳥")
        d = d.replace("烏", "鳥")
        d = d.replace("且", "目")
        d = d.replace("瓜", "爪")
        d = d.replace("令", "今")
        d = d.replace("老", "耂")
        d = d.replace("聿", "肀")
        d = d.replace("虎", "虍")
        d = d.replace("夬", "夫")
        d = d.replace("夬", "夫")
        d = d.replace("矢", "夫")
        d = d.replace("失", "夫")
        d = d.replace("必", "心")
        # d = d.replace("爰", "愛")
        # d = d.replace("主", "王")

        pattern1 = d[0]+ d[1] +"X"
        pattern2 = d[0]+ "X" + d[2]

        if d[0] == "⿱" and d[1] in "山木口土日":
            pattern1 = d[0]+ d[1]
        if d[0] == "⿱" and d[2] in "山木口土日":
            pattern2 = d[0]+ d[2]

        if d[0] == "⿰" and d[1] in "舌⺼風阝日土":
            pattern1 = d[0]+ d[1]
        if d[0] == "⿰" and d[2] in "舌⺼風阝日土":
            pattern2 = d[0]+ d[2]

        radicals = "耂肀虍光予军用多食耳正臣谷缶彐是豆林爪瓦鹿音角云共旦羽豕果里止厶見毛母比鬼青辛非文韋立子十弓方又牛今車魚夫羊巾田酉隹米禾|馬足貝山石王目火女鳥虫言"
        # The following candidates are not suitable for merging: 木口几

        # Classify the following as component, ignore composition
        if d[1] in radicals and radicals.index(d[1]) > -1:
            current = radicals[radicals.index(d[1])]
            if d[1] == "车":
                pattern1 = ("X車")
            elif d[1] == "鱼":
                pattern1 = ("X魚")
            else:
                pattern1 = ("X" + current)

        if d[2] in radicals and radicals.index(d[2]) > -1:
            current = radicals[radicals.index(d[2])]
            if d[2] == "车":
                pattern2 = ("X車")
            elif d[2] == "鱼":
                pattern2 = ("X魚")
            else:
                pattern2 = ("X" + current)

        if d[1] in "冖宀穴":
           pattern1 = "⿱冖X"

        if pattern2 == "⿰X戈":
            pattern2 = "⿰X弋"

        patterns.append(pattern1)
        patterns.append(pattern2)
    elif len(d) < 3 and "X" not in d:
        if "品" in d:
            patterns.append("品X")
        else:
            # 独体字
            patterns.append("口")

    for p in patterns:
        if p not in cat:
            cat[p] = [(idx,c)]
        else:
            cat[p].append((idx,c))
    # End of for loop

print("CAT:",len(cat))


for r in list(cat.keys()):

    if "?" not in r and "？" not in r and len(cat[r]) > 10 and r is not "X":
        # remove too detailed categories
        for idx, ch in cat[r] :
            if idx not in dic_c:
                dic_c[idx] = ch
    elif "正" in r or "父" in r or "龺" in r or "光" in r:
        # non frequent but common components
        for idx, ch in cat[r] :
            if idx not in dic_c:
                dic_c[idx] = ch
    else:
        del cat[r]

# reorganize data
for c_i,c in enumerate(cat):
    class2id[c] = c_i
    id2class[c_i] = c
    # os.makedirs("img/"+ c)
    for idx, char in cat[c]:
        if idx not in data:
            data[char2id[char]] = (str(c_i),char,c)
        else:
            x, y, z = data[char2id[char]]
            data[char2id[char]] = (x + " " + str(c_i), char, z + " " + c)

# print all labels
# for w in sorted(cat, key=lambda k: len(cat[k]), reverse=False):
#   print(w, len(cat[w]))

print("Categories:", len(cat))
# # print(dic_c)
print("Dictionary:",len(dic_c),"/", count)


# Records:
# Full v1: 295 categories 9574 characters
# Full v2: 251 categories 9049 characters (remove some simplified ch)
# Full v3: 250 categories 7852 characters (remove all X & radicales)
# Final:   256 categories 9140/9585 characters
# Final_v2:   256 categories 9416/9635 characters
# Final_v2.1:   256 categories 9474/9635 characters
# Final_v2.2: 256 categories 9476/9635 characters

# Format
# charid : type id, type in unicode

with open('full_' + str(len(cat)) + 'C.json', 'w') as outfile:
    json.dump(data, outfile)
    print("save to file")

writer = tsv.TsvWriter(open("v3_meta.tsv", "w"))
for idx, entry in enumerate(data):
    writer.line(str(idx) + "\t" + id2char[entry])
writer.close()
