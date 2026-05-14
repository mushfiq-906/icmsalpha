package com.mycompany.icmsalpha;

import java.util.Objects;

class SingleMethod {
    int revision;
    String methodName;
    String signature;
    String filePath;
    int startLine;
    int endLine;
    int methodId;
    int globalMethodId;
    String packageName = ""; // Java package name
    String className = ""; // Class or file name containing the method

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        SingleMethod that = (SingleMethod) o;
        return revision == that.revision &&
                Objects.equals(methodName, that.methodName) &&
                Objects.equals(signature, that.signature) &&
                Objects.equals(filePath, that.filePath);
    }

    @Override
    public int hashCode() {
        return Objects.hash(revision, methodName, signature, filePath);
    }
}
