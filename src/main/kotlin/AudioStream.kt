import org.apache.commons.math3.stat.descriptive.moment.Mean
import java.io.FileInputStream
import java.io.PrintStream
import java.util.*
import java.util.stream.Stream
import kotlin.streams.asStream

typealias AudioStream = ListIterator<Short>

fun Scanner.shortIterator() = ScannerIterator(this)

fun AudioStream.nextToken(): List<Short> {
    val list = mutableListOf<Short>()
    var input: Short
    input = next()
    val sign = (input < 0)
    list.add(input)
    while (hasNext()) {
        input = next()
        if (input != 0.toShort() && (input < 0) == sign)
            list.add(input)
        else {
            previous(); break
        }
    }
    return list
}

fun List<Short>.meanSample() = Pair(size, Mean().evaluate(this.stream().mapToDouble { it.toDouble() }.toArray()))

class ScannerIterator(private val scanner: Scanner) : ListIterator<Short> {

    var index = 0

    var last: Short = 0

    var useBuffer = false

    override fun hasPrevious(): Boolean = true

    override fun nextIndex(): Int = (index + 1)

    override fun previous(): Short {
        useBuffer = true
        index--
        return last
    }

    override fun previousIndex(): Int = (index - 1)

    override fun hasNext(): Boolean = scanner.hasNextShort()

    override fun next(): Short {
        if (useBuffer)
            useBuffer = false
        else {
            last = scanner.nextShort()
        }
        index++
        return last
    }

}

fun main(args: Array<String>) {
    val scan = Scanner(FileInputStream("D:\\zwz.txt"))
    System.setOut(PrintStream("D:\\zwz2.txt"))
    var index = 0
    scan.shortIterator().apply {
        while (hasNext()) nextToken().apply {
            if (size > 2)
                println("" + (index + (size / 2)) + ',' + Mean().evaluate(this.stream().mapToDouble { it.toDouble() }.toArray()))
            index += size
        }
    }
}
