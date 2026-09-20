"""Render a Chinese illustrated PDF using existing ReportLab/Pillow runtime."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data/ranked-v1"
OUT = ROOT/"output/pdf/font-recognition-report.pdf"
WIDTH,HEIGHT = 595.28,841.89
INK = HexColor("#172936")
ACCENT = HexColor("#087E8B")
MUTED = HexColor("#58707E")


def main():
    pdfmetrics.registerFont(TTFont("CJK","/System/Library/Fonts/STHeiti Light.ttc",subfontIndex=0))
    data = json.loads((DATA/"results.json").read_text())
    rows = data["results"]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    c = canvas.Canvas(str(OUT),pagesize=(WIDTH,HEIGHT))
    c.setTitle("日文漫画字体识别 - 字族候选与置信参考报告")
    c.setAuthor("MikanFontProbe")
    page = 0

    def text(x,y,value,size=11,color=INK):
        c.setFillColor(color)
        c.setFont("CJK",size)
        c.drawString(x,y,str(value))

    def paragraph(value,x,y,w=515,size=11):
        style = ParagraphStyle("body",fontName="CJK",fontSize=size,leading=size*1.6,textColor=INK,wordWrap="CJK")
        p = Paragraph(escape(value).replace("\n","<br/>"),style)
        _,h = p.wrap(w,700)
        if y-h<48:
            raise ValueError("Report text overflow")
        p.drawOn(c,x,y-h)
        return y-h

    def start(title,kicker="MIKAN FONT PROBE / LOCAL EXPERIMENT"):
        nonlocal page
        page += 1
        c.setFillColor(ACCENT)
        c.rect(40,795,42,4,fill=1,stroke=0)
        text(40,813,kicker,8,MUTED)
        text(40,760,title,23)
        text(40,28,"实验报告 · 参考置信度不是正确概率 · 字体和原图仅本地处理",8,MUTED)
        text(533,28,str(page),9,MUTED)

    def picture(path,x,y,max_w,max_h):
        with Image.open(path) as image:
            w,h = image.size
        ratio = min(max_w/w,max_h/h)
        c.drawImage(str(path),x,y-h*ratio,width=w*ratio,height=h*ratio,mask="auto")
        return h*ratio

    start("字体候选，不再只给一个答案")
    y = paragraph("本轮增加字族去重、输入质量加权、多字一致性和移除单字后的稳定性检查，并为前三候选给出 0-100 的参考置信度。所有分数均未做概率校准，不表示正确率。",40,724)
    cards = [("23", "字体文件"),("14", "独立候选类别"),("21", "复用真实截图")]
    for i,(number,label) in enumerate(cards):
        x = 40+i*178
        c.setFillColor(HexColor("#EEF5F5"));c.roundRect(x,564,158,90,8,fill=1,stroke=0)
        text(x+16,609,number,27,ACCENT);text(x+16,584,label,11)
    y = paragraph("核心规则\n同一字族不同字重合并；Antique AN+ 与攸望分别作为候选，但都映射黑体。RodinHappy L / B / UB 是例外，分别保留为少女 / 方圆 / 海报。候选按图形距离排序，不按参考置信度排序。",40,537)
    text(40,y-35,"本轮结果：必须分清候选范围",15)
    y = paragraph("原三类字库（12 款、4 个日文字族）：开发样本 7/9，原留出样本 11/12。\n全部字库（23 款、14 个类别）：开发样本 6/9，原留出样本 8/12。\n质量加权在本批样本上没有改变 Top-1 正确数；扩大字库后相似候选增加，原先的 11/12 不能沿用为全字库表现。",40,y-55)
    y = paragraph("这次的改进主要是更透明的候选与风险提示，不是已证明的准确率提高。21 张图中 18 张触发复核提示。过去的留出集已被反复观察，本轮结果只能称为复用样本回归，不是新的盲测。",40,y-22)
    paragraph("范围：仍使用人工核对的文字及既有裁片；没有接入自动 OCR、整页定位或自动补采更多字符。空的角明体、降圆体目录不在候选库。",40,y-24,size=10)
    c.showPage()

    start("全样本总览")
    paragraph("下表第一候选来自全部 14 类候选。参考分低不等于已知错误，高也不保证正确；题库答案用于事后对照，未输入匹配与置信计算。",40,722,size=10)
    headers=[(40,"题目"),(91,"题库大类"),(190,"第一候选"),(395,"参考分"),(465,"答案一致")]
    for x,label in headers:text(x,653,label,10,MUTED)
    y = 627
    for i,r in enumerate(rows):
        if i%2==0:
            c.setFillColor(HexColor("#F2F6F7"));c.rect(36,y-8,522,23,fill=1,stroke=0)
        t = r["top3"][0]
        text(40,y,"q"+str(r["qid"]),10);text(91,y,r["expected"],10)
        text(190,y,t["name"],10);text(395,y,f"{t['confidence']:.1f}/100",10)
        text(465,y,"是" if t["answer"]==r["expected"] else "否",10)
        y -= 24
    paragraph("两批样本不合并成泛化准确率；人工裁片不等于自动切割验收。下一页解释分数，再用六个案例展示原图和同字字体对照。",40,94,size=10)
    c.showPage()

    start("参考置信度如何产生")
    blocks = [
        ("1. 输入质量：不参考答案或候选", "按原始分辨率、对比度、墨迹触边风险估计质量。空白或全黑拒绝；质量低于 0.25 不参与。最多选 5 个候选字。本轮每图仅已有 3 字，因此还没有验证从大量字符自动挑字的收益。"),
        ("2. 多字匹配：不拼凑最佳字重", "每个字与每款字体同字比较，使用上一轮固定的混合距离。按输入质量加权求平均，再取每个字族的最佳款式。同一候选必须用同一款字体解释所有入选字符。坏字不因为不支持第一名就被删除。"),
        ("3. 一致性与稳定性", "统计各字单独支持哪个字族，再逐个移除字符观察第一候选是否变化。不同字符意见冲突、前两名距离过近或库内最佳距离仍很大，都会提示复核。"),
        ("4. 参考分不是 softmax 概率", "分数综合绝对拟合、平均质量、字级支持、移除单字稳定性、前两名分差。各候选分数不要求相加为 100；前三名仍按匹配距离排序。0-100 仅是可解释证据指数，阈值为初始规则，尚未校准。"),
        ("5. 使用边界", "不足 3 字时参考分上限 45；其他情况上限 90。分差小于 0.02、支持或稳定性低于 0.67、参考分低于 40 等触发复核。系统不会因为没有提示就自动通过；没有验证过真正的缺库拒判率。")]
    y=720
    for title,body in blocks:
        text(40,y,title,13,ACCENT)
        y=paragraph(body,40,y-15,size=10)-29
    paragraph("验证：28 项回归测试通过，涵盖字重合并例外、空白拒绝、触边降权、同款多字聚合、低分差提示，以及冻结基线完整性。旧评分源码与留出输入未修改。",40,y,size=10)
    c.showPage()

    notes = {
        104:"攸望日文字形成为首选。不同字重没有重复占据前三位置；触边提示仍保留，不因匹配较好就忽略输入风险。",
        114:"Ryumin 的多个字重合并为一个候选。多个字支持较一致，但参考分依然不是已校准正确概率。",
        98:"反白原图保留在左侧，匹配输入预先翻转极性。候选图使用黑字白底，便于比较字形；这不是修改漫画原件。",
        129:"三类候选时混合评分正确；全字库加入 RodinHappy 后首选发生变化。低分差与多字冲突提示能暴露这种不确定性，不能继续宣称此题已解决。",
        99:"这是持续存在的错误：新丸ゴ第一、题库答案为黑体。系统提供候选与复核理由，但尚不能判断是字体缺库、源图差异还是评分偏差。",
        72:"加入 Comic Mystery 后它超过 Ryumin，导致全字库分类错误。宋体仍在前三。说明仅在圆/宋/黑三个大类内验证不足以代表全字库性能。"}
    for qid in (104,114,98,129,99,72):
        r = next(r for r in rows if r["qid"]==qid)
        start(f"q{qid} · 原图与前三字体")
        text(40,724,f"题库答案：{r['expected']}  |  状态：{r['status']}",10)
        text(40,699,"原图（橙框为所用字符）",10,MUTED)
        picture(DATA/"evidence"/f"q{qid}-source.png",40,684,174,304)
        text(40,357,"实际匹配输入",10,MUTED)
        for i,g in enumerate(r["glyphs"]):
            picture(ROOT/g["path"],40+i*59,338,52,62)
            text(40+i*59,260,g["character"],12)
            text(40+i*59,244,f"Q {g['quality']['score']:.2f}",8,MUTED)
        for rank,t in enumerate(r["top3"]):
            y=693-rank*148
            text(240,y,f"{rank+1}. {t['name']}",13,ACCENT)
            text(240,y-21,f"中文：{t['answer']}   参考置信度：{t['confidence']:.1f}/100",9)
            text(240,y-39,f"距离 {t['distance']:.4f}  |  {t['representative_face']}",7.5,MUTED)
            picture(DATA/"evidence"/f"q{qid}-candidate{rank+1}.png",240,y-49,305,91)
        y=paragraph(notes[qid],40,204,size=10)
        paragraph("复核原因："+("；".join(r["warnings"]) or "未触发规则警告，仍需人工核对"),40,y-15,size=10)
        c.showPage()

    start("交付内容与下一验证边界")
    y=paragraph("本轮代码提供：字族映射、全字库模板、质量加权排名、前三候选、未校准置信参考和复核理由。所有原始字体与漫画原图保持不变；没有把字体或图片发送到外部服务。",40,719)
    text(40,y-38,"字族合并清单",14,ACCENT)
    families = {}
    for f in data["fonts"]:
        g=f["family_group"]
        families.setdefault(g["key"],{"name":g["name"],"answer":g["answer"],"count":0})["count"]+=1
    y-=65
    for i,f in enumerate(families.values()):
        text(40,y,f"{f['name']}  →  {f['answer']}  /  {f['count']} 款",10)
        y-=21
    y=paragraph("主要文件\ndata/ranked-v1/results.json：21 题的前三候选、质量和复核证据。\ndata/ranked-v1/templates/catalog.json：字族、文件指纹和模板。\nrank_font_candidates.py：评分与参考置信计算。\nfont_family_policy.py：字重合并与 RodinHappy 例外。",40,y-14,size=10)
    paragraph("下一验证重点：新增每块多个可读字符验证自动选字；准备真正缺库与近似字体负例，校准拒判和参考分；在不复用现有样本调参的前提下做全字库验证。当前数据不足以输出可靠的“正确概率”。",40,y-24,size=10)
    c.showPage()
    for offset in (0,11):
        start("附录 · 全样本前三候选")
        paragraph("按匹配距离排序；每个候选下方为中文答案及未校准参考分。不同日文字族可对应同一个中文答案，字重不会重复占位（RodinHappy 例外）。",40,722,size=10)
        for x,label in ((40,"题目"),(92,"第一候选"),(248,"第二候选"),(404,"第三候选")):
            text(x,656,label,10,MUTED)
        y=628
        for index,r in enumerate(rows[offset:offset+11]):
            if index%2==0:
                c.setFillColor(HexColor("#F2F6F7"));c.rect(36,y-29,522,48,fill=1,stroke=0)
            text(40,y,"q"+str(r["qid"]),10)
            for col,t in enumerate(r["top3"]):
                x=92+col*156
                text(x,y,t["name"],9)
                text(x,y-18,f"{t['answer']} / {t['confidence']:.1f}分",8,MUTED)
            y-=49
        c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
