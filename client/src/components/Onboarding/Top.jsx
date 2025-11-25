import React, { useEffect, useState } from "react";
import { sliderImages } from "../../assets/assets";
import { MoveRight } from "lucide-react";
import { Link } from "react-router-dom";
import { useAppContext } from "../../context/AppContext";

const Top = () => {
  const { navigate } = useAppContext();
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrent((prev) => (prev + 1) % sliderImages.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  const prevIndex = (current - 1 + sliderImages.length) % sliderImages.length;
  const nextIndex = (current + 1) % sliderImages.length;

  return (
    <section className="mb-20 text-center text-black">
      <h2 className="text-sm uppercase tracking-wide mb-4 mt-12 font-bold text-special-red">
        MRI Tumor Detection System
      </h2>

      <h1 className="text-2xl md:text-5xl max-w-4xl mx-auto font-light mb-2 md:mb-10 px-4 leading-snug">
        Your trusted deep learning brain tumor detection model, scan your report
      </h1>

      {/* Image Slider */}
      <div className="relative w-full h-72 flex items-center justify-center overflow-hidden">
        <div className="relative w-full h-full flex items-center justify-center">
          {/* Previous Image */}
          <img
            src={sliderImages[prevIndex]}
            alt="previous"
            className="absolute left-50 w-72 opacity-70 rounded-lg object-cover shadow-lg transform transition-transform duration-700"
          />
          {/* Current Image */}
          <img
            src={sliderImages[current]}
            alt="current"
            className="absolute w-96 z-10 rounded-lg object-cover shadow-xl transform transition-transform duration-700 scale-105"
          />
          {/* Next Image */}
          <img
            src={sliderImages[nextIndex]}
            alt="next"
            className="absolute right-50 w-72 opacity-70 rounded-lg object-cover shadow-lg transform transition-transform duration-700"
          />
        </div>
      </div>

      {/* Button */}
      <div onClick={() => { navigate('/tools'); scrollTo(0, 0) }}
        className="inline-flex items-center justify-center px-6 py-3 bg-special-red/10 text-black rounded-full hover:bg-special-red/15 shadow-sm transition mt-2 md:mt-10 group cursor-pointer"
      >
        Scan Your Report
        <span className="ml-3 inline-flex items-center justify-center w-6 h-6 rounded-full text-sm group-hover:translate-x-2 transition-all duration-500">
          <MoveRight
            size={20}
            className="-rotate-45 group-hover:rotate-0 transition-all duration-500"
          />
        </span>
      </div>
    </section>
  );
};

export default Top;
