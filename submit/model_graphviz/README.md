# graphviz 教程

graphviz是一个开源的图形可视化软件，与其他图形软件所不同，它的理念是“所想即所得”，通过dot语言来描述并绘制图形。

安装graphviz 

```shell
sudo apt-get install graphviz graphviz-doc
```

这样会安装dot语言的执行文件，执行文件路径在

```shell
/usr/bin/dot
```

绘图方法

先根据业务需要编写dot文件，参见下面的示例部分，然后编译运行。输出格式可以根据自己的需要来灵活选择
例如test.dot, 产生图片：

```shell
dot -Tpng test.dot -o test.png
dot -Tsvg test.dot -o test.svg
dot test.dot -Tpdf -o test.pdf
```
