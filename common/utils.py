import xlwt


# 保存 loss and accuracy 的值
def save_metrics(path, train_loss, test_loss, acc):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    names = ['train_loss', 'test_loss', 'acc']
    for j in range(len(names)):
        sheet1.write(0, j, names[j])
        for i in range(len(acc)):
            sheet1.write(i+1, j, eval(names[j])[i])
    f.save(path+'metircs.xlsx')
