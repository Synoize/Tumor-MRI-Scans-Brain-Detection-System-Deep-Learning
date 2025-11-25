
import React from 'react';
import { useAppContext } from '../context/AppContext';
import Top from '../components/Onboarding/Top';
import Solutions from '../components/Onboarding/Solutions';
import About from '../components/Onboarding/About';
import Industries from '../components/Onboarding/Industries';
import Resources from '../components/Onboarding/Resources';
import CaseStudies from '../components/Onboarding/CaseStudies';
import Blogs from '../components/Onboarding/Blogs';
import FAQ from '../components/Onboarding/FAQ';
import RequestDonation from '../components/Onboarding/RequestDonation';

const OnboardingScreen = () => {

    const { setShowUserLogin } = useAppContext()
    return (
        <div>
            <Top/>
            <Solutions/>
            <About/>
            <Industries/>
            <CaseStudies/>
            <Resources/>
            <Blogs/>
            <FAQ/>
            <RequestDonation/>
        </div>
    );
};

export default OnboardingScreen
