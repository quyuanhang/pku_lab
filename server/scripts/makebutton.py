s = str()
for y in range(1901, 2016):
    s += ('<li><a href="javascript:year_filter(%d)">%d</a></li>' % (y, y))