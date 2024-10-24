import qbs

CppApplication {
    consoleApplication: true
    files: "main.cpp"

    Group {     // Properties for the produced executable
        fileTagsFilter: "application"
        qbs.install: true
        qbs.installDir: "bin"
    }
}
