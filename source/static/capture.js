/**
Handles client-side interaction, primarily the photoshoot.

@author: Elias Gabriel, Duncan Mazza
@revision: v1.0
**/
const ENDPOINT = '/capture';

document.body.onkeyup = function(e) {
    if(e.keyCode == 32) {
		let canvas = document.createElement('canvas');
		let im = document.getElementById("capture");
		let bb = im.getBoundingClientRect();
		canvas.width = bb.width;
		canvas.height = bb.height;

		let context = canvas.getContext('2d');
		context.drawImage(im, 0, 0, bb.width, bb.height);
	
		let data = new FormData();
		data.append('URI', canvas.toDataURL('image/jpeg'));
		fetch(ENDPOINT, {
			method: "POST",
			body: data
		}).then(() => window.location.replace('/show'));
	}
};
