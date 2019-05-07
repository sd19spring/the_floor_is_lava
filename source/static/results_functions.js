function createDivs(numCaps) {
    // Dynamically creates the html elements used to display the camera feeds - numCaps defines how many cameras there
    // are to display.
    let i;  // for iteration
    for (i = numCaps; i >= 0; i--) {
        // Create a new, plain <span> element
        var frame = document.createElement("div");
        var eye_str = '/' + i + 'heatmap';  // set the url used for the source of the image
        frame.setAttribute('class', 'capture_div');
        frame.innerHTML = "<img alt='[loading heatmap...]' src=" + eye_str + " class='capture'/>"
        // Get a reference to the element, before we want to insert the element
        var sp2 = document.getElementById("childElement");
        // Get a reference to the parent element
        var parentDiv = sp2.parentNode;
        // Insert the new element into the DOM before sp2
        parentDiv.insertBefore(frame, sp2.nextSibling);
    }
}

function incrementHeatmapDisplayDown() {

    $.post("/increment_heatmap_display", 'down');
    // stop link reloading the page
    event.preventDefault();
}

function incrementHeatmapDisplayUp() {
    $.post("/increment_heatmap_display", 'up');
    // stop link reloading the page
    event.preventDefault();
}

function turnOffCameras() {
    // ajax the JSON to the server
    $.post('/all_cam_switch', {0: 0});
    // stop link reloading the page
    event.preventDefault();
}