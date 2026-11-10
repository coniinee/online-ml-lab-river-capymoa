package lab4;

import java.nio.file.*;
import java.util.*;
import java.util.jar.*;
import jdk.internal.org.objectweb.asm.*;

/** Produce an overlay; never alter the installed MOA distribution. */
public final class PatchAutoML {
    public static void main(String[] args) throws Exception {
        Path output = Path.of(args[1]);
        Files.createDirectories(output);
        try (JarFile jar = new JarFile(args[0])) {
            for (JarEntry entry : Collections.list(jar.entries())) {
                String name = entry.getName();
                if (!name.startsWith("moa/classifiers/meta/AutoML/") || !name.endsWith(".class")) continue;
                byte[] bytes = jar.getInputStream(entry).readAllBytes();
                ClassReader reader = new ClassReader(bytes);
                ClassWriter writer = new ClassWriter(reader, ClassWriter.COMPUTE_MAXS);
                int[] replacements = {0};
                reader.accept(new ClassVisitor(Opcodes.ASM7, writer) {
                    @Override public MethodVisitor visitMethod(int access, String method, String desc, String sig, String[] exceptions) {
                        return new MethodVisitor(Opcodes.ASM7, super.visitMethod(access, method, desc, sig, exceptions)) {
                            @Override public void visitMethodInsn(int opcode, String owner, String method, String desc, boolean itf) {
                                if (opcode == Opcodes.INVOKESTATIC && owner.equals("java/lang/Math") && method.equals("random") && desc.equals("()D")) {
                                    super.visitMethodInsn(opcode, "lab4/SeededRandom", method, desc, false);
                                    replacements[0]++;
                                } else if (opcode == Opcodes.INVOKESPECIAL && owner.equals("java/util/Random") && method.equals("<init>") && desc.equals("()V")) {
                                    super.visitMethodInsn(Opcodes.INVOKESTATIC, "lab4/SeededRandom", "nextSeed", "()J", false);
                                    super.visitMethodInsn(opcode, owner, method, "(J)V", false);
                                    replacements[0]++;
                                } else super.visitMethodInsn(opcode, owner, method, desc, itf);
                            }
                        };
                    }
                }, 0);
                if (replacements[0] > 0) {
                    Path target = output.resolve(name);
                    Files.createDirectories(target.getParent());
                    Files.write(target, writer.toByteArray());
                    System.out.println(name + " " + replacements[0]);
                }
            }
        }
    }
}
