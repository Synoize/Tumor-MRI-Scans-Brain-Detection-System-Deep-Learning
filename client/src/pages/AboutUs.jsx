import React from "react";
import { useAppContext } from "../context/AppContext";
import { assets } from "../assets/assets";

const AboutUs = () => {
  const { navigate } = useAppContext();

  return (
    <section className="container mx-auto mt-12 md:mb-20 mb-8 px-4 md:px-12 flex flex-col lg:flex-row items-start text-black">
      {/* Left: Image */}
      <div className="w-full lg:w-1/2">
        <img
          loading="lazy"
          src={assets.about}
          alt="About"
          className="w-full h-auto rounded-2xl object-cover"
        />
      </div>

      {/* Right: Content */}
      <div className="w-full lg:w-1/2 space-y-4 px-4 py-12 md:px-12">
        <h3 className="text-sm font-bold uppercase tracking-wide text-special-red">About Us</h3>
        <h2 className="text-2xl md:text-5xl font-light leading-snug">
          Brain tumor detection has been a critical area of
        </h2>
        <p className="font-light md:text-xl">
          research in medical imaging due to the complexity and severity of brain
          tumors. Manual diagnosis using MRI scans is time-consuming and prone to
          human error.
        </p>

      </div>
    </section>
  );
};

export default AboutUs;
