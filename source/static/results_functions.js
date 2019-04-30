function createDivs(numCaps) {
    // Dynamically creates the html elements used to display the camera feeds - numCaps defines how many cameras there
    // are to display.
    let i;  // for iteration
    for (i = numCaps; i >= 0; i--) {
        // Create a new, plain <span> element
        var frame = document.createElement("div");
        var eye_str = '/' + i + 'heatmap';
        frame.setAttribute('class', 'capture_div');
        frame.innerHTML = "<img alt='[loading heatmap...]' src=" + eye_str + " class='capture'/>"
        // Get a reference to the element, before we want to insert the element
        var sp2 = document.getElementById("childElement");
        // Get a reference to the parent element
        var parentDiv = sp2.parentNode;
        // Insert the new element into the DOM before sp2
        parentDiv.insertBefore(frame, sp2.nextSibling);

        muteDict[i] = 1;  // 1 indicates that the camera should be turned on; 0 indicates that it shouldn't
        calibDict[i] = 1;  // 1 indicates that the camera is not using calibration; 0 indicates that it is
    }
}