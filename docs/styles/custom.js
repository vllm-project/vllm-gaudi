// Remove the default box shadow under the top navigation bar when the page isn't scrolled
document.addEventListener("DOMContentLoaded", () => {
  const header = document.querySelector(".md-header");
  if (!header) return;

  const updateHeaderShadow = () => {
    if (window.scrollY === 0) {
      header.style.boxShadow = "none";
      header.style.borderBottom = "1px solid #0000001a";
    } else {
      header.style.boxShadow = "0 0 .2rem #0000001a, 0 .2rem .4rem #0003";
      header.style.borderBottom = "none";
    }
  };

  window.addEventListener("load", updateHeaderShadow);
  window.addEventListener("scroll", updateHeaderShadow);
});
