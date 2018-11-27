import org.apache.commons.math3.stat.descriptive.moment.Mean
import java.awt.Dimension
import java.awt.Graphics
import java.io.FileOutputStream
import java.io.PrintStream
import java.nio.ByteBuffer
import java.util.*
import javax.sound.sampled.*
import javax.swing.JFrame
import javax.swing.JPanel

const val frameRate = 44100

val defaultFormat =
        AudioFormat(frameRate.toFloat(), 16, 1, true, true)

val audioInput: MutableList<Int> = Collections.synchronizedList(mutableListOf())

class PaintCanvas : JPanel() {
    override fun paint(g: Graphics?) {
        super.paint(g!!)
        audioInput.forEachIndexed { index: Int, i: Int ->
            println(i)
            g.fillRect((index * 2), 300, 1, (i) / 20 + 1)
        }
        var index = 0
        if (input!!.available() > 1) {
            if (audioInput.size >= 500) audioInput.clear()
            val arr = ByteArray(32000)
            val size = input?.read(arr)
            ByteBuffer.wrap(arr).asShortBuffer().apply {
                0.rangeTo(size!!).forEach {
                    audioInput.add(it)
                }
            }
        }
    }
}

val canvas = PaintCanvas()

private val line: TargetDataLine = AudioSystem.getTargetDataLine(defaultFormat)
var input: AudioInputStream? = null

fun start() {
    var index = 0
    val input = AudioInputStream(line)
    while (true)
        if (input.available() > 0) {
            if (audioInput.size == 1000) audioInput.clear()
            audioInput += (input.read().toByte()).toInt()
            canvas.repaint()
        }
}


fun main(args: Array<String>) {
    val p = PrintStream(FileOutputStream("D:\\wxx.txt"), true)
    line.open()
    line.start()
    val input = AudioInputStream(line)
    /*JFrame().apply {
        size = Dimension(1200, 720)
        contentPane = canvas
        isVisible = true
    }*/
    val arr = ByteArray(frameRate)
    var tm = 0
    while (true) {
        val size = input.read(arr) / 2
        ByteBuffer.wrap(arr).asShortBuffer().apply {
            (0 until size).forEach {
                p.println("${this[it]}")
            }
        }
        println("+ 1 second")
        if (tm++ >= 10) break
    }
}
