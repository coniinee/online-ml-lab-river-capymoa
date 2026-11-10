package lab4;

import java.util.Random;

/** The single explicit random source used by the patched AutoML package. */
public final class SeededRandom {
    private static Random source = new Random(0L);
    private SeededRandom() {}
    public static synchronized void reset(long seed) { source = new Random(seed); }
    public static synchronized void use(Random random) { source = random; }
    public static synchronized double random() { return source.nextDouble(); }
    public static synchronized long nextSeed() { return source.nextLong(); }
}
