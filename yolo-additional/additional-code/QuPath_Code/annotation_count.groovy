// QuPath script: Count annotations per file
// Works in QuPath v0.3+ (Groovy)

// Get current project
def project = getProject()
if (project == null) {
    print 'No project open!'
    return
}

// Iterate over all image entries in the project
for (entry in project.getImageList()) {
    def server = entry.readImageData().getServer()
    def hierarchy = entry.readHierarchy()
    def annotations = hierarchy.getAnnotationObjects()

    // Count annotations by class name
    def counts = [:].withDefault { 0 }
    for (ann in annotations) {
        def name = ann.getPathClass() != null ? ann.getPathClass().getName() : 'Unclassified'
        counts[name] += 1
    }

    // Print file name
    println "File: ${entry.getImageName()}"
    counts.each { cls, num ->
        println "    ${cls}: ${num}"
    }
    println ""
}
