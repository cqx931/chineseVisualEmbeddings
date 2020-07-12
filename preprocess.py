# -*- coding: utf-8 -*-
"""
@author: cqx931
2020
"""
import os
import json
import pygame
from pygame import ftfont
import pickle
import operator
import csv
import tsv
import re
import sys

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode

# based on the csv, all characters and its id
char2id = {}
id2char = {}
# a dictionary that contains all the categories and its characters
cat = {}
# a compact dictionary that contains all characters
dic_c = {}

# all classes
class2id = {}
id2class = {}
# final data format for training
data = {}

SIZE = 32
LIMIT = 11
debug = True

COMP  = ["⿱","⿳", "⿰","⿲", "⿻","⿸","⿹", "⿺", "⿷", "⿴", "⿵","⿶"]

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

def drawImages():
    pygame.init()
    # same image size, better quality
    dict = open("data/VC/dict.csv", "r")
    c_image = 0
    missing = []

    if not os.path.exists('data/VC/img_all/'):
        os.makedirs('data/VC/img_all/')

    for idx,entry in enumerate(dict):
        items = entry.split(',')
        unicode = items[2].replace("\n","")
        valid = draw(items[0], idx, "data/VC/img_all/")
        if valid:
            c_image +=1
        else:
            missing.append(idx)
        if idx%100 == 0:
            print(idx)
    print("Total images generated:", c_image)
    print("Missing:", missing)

# create a clear csv file from the original dict file
def generateCSV():
    raw_dict = open("data/VC/dict_fix.csv", "r")
    with open('data/VC/dict.csv', 'w') as csvfile:
        fieldnames = ['character', 'decomposition','unicode']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        for idx,entry in enumerate(raw_dict):
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

            if (c == r): d = 'X'

            if len(d) == 4:
                if '⿳' in d:
                    d = '⿱' + d[1] + d[3]
                elif '⿲' in d:
                    d = '⿰' + d[1] + d[3]
            elif len(d) > 4 and d[1] not in COMP:
                d = d[0] + d[1] + "?"
            elif len(d) > 4:
                d = d[0] + "?" + d[len(d)-1]
            {'character': c, 'decomposition': d, 'radical':r}

def simplifyDecomposition(d):
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
    d = d.replace("仑", "侖")
    d = d.replace("仓", "倉")
    d = d.replace("乔", "喬")
    d = d.replace("区", "區")
    d = d.replace("夹", "夾")
    d = d.replace("会", "會")
    d = d.replace("争", "爭")
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
    d = d.replace("龺", "𠦝")
    d = d.replace("⺼", "月")
    d = d.replace("攵","夂")
    d = d.replace("𠂊", "⺈")
    # d = d.replace("爰", "愛")
    d = d.replace("主", "王")
    return d

def getMostCommonCh():
    mostCommonChs = ''
    basic_dict = open("data/VC/mostCommon.txt", "r")
    for line in basic_dict:
        mostCommonChs = line
    return mostCommonChs.replace("\n","")

def getRefCSV():
    ref = {}
    with open('data/VC/dict_fix.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            ref[rows[0]] = rows[1]
    return ref

def addToDict(item, dict, value):
    if item not in dict:
        dict[item] = [value]
    else:
        dict[item].append(value)

def getPatterns(c,d):
    patterns = []
    # unclassifiable common ch
    if c in "幻巧枣垂甚临既释就疏聚舞疑敲黎氓衅棘甥粤赫兢孵幾虧歸叢務發執鞏喬壯農報縣畝棗隸參斃競營釋賴疊釁粵噩鹵":
        patterns.append("口")

    if len(d) == 3 and "X" not in d:
        d = simplifyDecomposition(d);

        pattern1 = d[0]+ d[1] +"X"
        pattern2 = d[0]+ "X" + d[2]

        if d[0] == "⿱" and d[1] in "山木口土日頁":
            pattern1 = d[0]+ d[1]
        if d[0] == "⿱" and d[2] in "山木口土日頁":
            pattern2 = d[0]+ d[2]

        if d[0] == "⿰" and d[1] in "舌月風阝日土":
            pattern1 = d[0]+ d[1]
        if d[0] == "⿰" and d[2] in "舌月風阝日土":
            pattern2 = d[0]+ d[2]

        if "⿻行" in d:
            pattern1 = "⿰彳X"

        radicals = "耂肀虍光予军用多食耳正臣谷缶彐是豆林爪瓦鹿音角云共旦羽豕果里止厶見毛母比鬼青辛非文韋立子十弓方又牛今車魚夫羊巾田酉隹米禾|馬足貝山石王目火女鳥虫言父儿麥尚艹咸直"
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
    else:
        patterns.append("X")
    return patterns

def addAllChToDict(all):
    for idx, ch in all :
        if idx not in dic_c:
            dic_c[idx] = ch

def addMissingCommonCh():
    allCh = "".join(dic_c.values())
    mostCommonChs = getMostCommonCh();
    ref_dict = getRefCSV()
    tobeAdded = 0
    for c in list(mostCommonChs):
        if c not in allCh:
            solved = False
            d = ref_dict[c]
            idx = char2id[c]
            patterns = getPatterns(c, d)
            if "口" in patterns or "X" in patterns:
                addToDict("口", cat, (idx,c))
                dic_c[idx] = c
            else:
                for p in patterns:
                    if p in cat:
                        cat[p].append((idx,c))
                        solved = True
                        # break patterns loop
                        break
                if not solved:
                    tobeAdded += 1
                    print(c, d, patterns)
    print("Still need to add:", tobeAdded)

def renderData():
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

def postProcess():
    count = 0
    csv = open('data/VC/dict_fix.csv', "r")
    for idx,entry in enumerate(csv):
        count += 1
        entry = entry.split(',')
        c = entry[0]
        d = entry[1]
        id2char[idx] = c
        char2id[c] = idx
        patterns = getPatterns(c, d)
        for p in patterns:
            addToDict(p, cat, (idx,c))
        # End of for loop

    print("All Categories:",len(cat))

    # clearn cat{} & render dict_c
    for r in list(cat.keys()):
        if "?" not in r and "？" not in r and len(cat[r]) > LIMIT and r is not "X" and r not in ["⿰镸X","⿱X廾","⿱殸"]:
            # remove too detailed categories
            addAllChToDict(cat[r])
        elif r in ["X父","⿺廴X","⿰𠦝X","⿱吅X","X麥","X正","X光","⿱X天","⿱X蟲","X直"] :
            # non frequent but common components
            addAllChToDict(cat[r])
        elif r in ["⿰X反","X尚","⿱X八","X用","⿱覀X","⿱X示","⿱臼X","⿱X夂","⿱X小","⿸户X","X彐","⿰X卜","⿱X寸","⿱丷X","⿱龹X","⿶凵X","⿵几X", "⿵冂X","⿰赤X","X予","X多","X食","⿱X几","X臣","X咸","⿰X佥"]:
            addAllChToDict(cat[r])
        else:
            del cat[r]
            # if debug:
            #     print("Delete category:", r)

    addMissingCommonCh()
    renderData()

    # print all final labels
    if debug:
        for w in sorted(cat, key=lambda k: len(cat[k]), reverse=True):
            print(w, len(cat[w]))

    print("Final Categories:", len(cat))
    # # print(dic_c)
    print("Dictionary:",len(dic_c),"/", count)
    name = "v3.2"

    with open('data/VC/'+ name +'/charset_' + str(len(cat)) + '.txt', 'w') as outfile:
        allCh = "".join(dic_c.values())
        outfile.write(allCh)
        print("charset: save to file")

    with open('data/VC/' + name +'/' + name + "_" +  str(len(cat)) + 'C.json', 'w') as outfile:
        json.dump(data, outfile)
        print("json:save to file")

    # writer = tsv.TsvWriter(open("data/VC/"+ name +"/" + name +"_meta.tsv", "w"))
    # for idx, entry in enumerate(data):
    #     writer.line(str(idx) + "\t" + id2char[entry])
    # writer.close()
    # print("tsv:save to file")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-img":
        generateCSV()
        drawImages()
    else:
        generateCSV()
        postProcess()

if __name__ == '__main__':
    main()

# 9484 256C
